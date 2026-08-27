"""
benchmark.py — Day 5/6: MLflow + W&B tracking, precision@3 rerank lift, clean latency
"""
import os
import time
import json
import pickle
from pathlib import Path
import yaml
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def cosine_sim(a, b):
    a = np.array(a); b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

def run_benchmark(provider="groq", limit=25):
    cfg = load_config()
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"[bench] provider={provider} device={device}")

    eval_path = Path(cfg["benchmark"]["eval_queries_path"])
    queries = [l.strip() for l in eval_path.read_text(encoding="utf-8").splitlines() if l.strip()][:limit]

    # load labels for precision@3
    labels_path = Path("data/processed/eval_labels.json")
    if not labels_path.exists():
        labels_path = Path(cfg["paths"]["processed_dir"]) / "eval_labels.json"
    eval_labels = {}
    if labels_path.exists():
        eval_labels = json.loads(labels_path.read_text(encoding="utf-8"))
        print(f"[bench] loaded labels for {len(eval_labels)} queries from {labels_path}")
    else:
        print(f"[warn] no eval_labels.json at {labels_path}, precision will be 0")

    import mlflow
    mlflow.set_experiment(cfg["benchmark"]["mlflow_experiment"])

    wandb_run = None
    try:
        import wandb
        if os.getenv("WANDB_API_KEY"):
            wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb_run = wandb.init(project=cfg["benchmark"]["wandb_project"], name=f"bench-{provider}-{int(time.time())}", reinit=True)
    except Exception as e:
        print(f"[wandb] init failed: {e}")

    # load embedding model for relevance scoring (transformers, avoid ST segfault)
    from transformers import AutoTokenizer, AutoModel
    import torch
    cache_path = Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    load_name = str(cache_path) if cache_path.exists() else cfg["models"]["embedding"]
    tok = AutoTokenizer.from_pretrained(load_name, local_files_only=cache_path.exists())
    mdl = AutoModel.from_pretrained(load_name, local_files_only=cache_path.exists()).to(device)
    mdl.eval()
    def _emb(texts):
        with torch.no_grad():
            out_all = []
            for i in range(0, len(texts), 16):
                batch = texts[i:i+16]
                inputs = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out = mdl(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                out_all.append(emb.cpu().numpy())
            return np.vstack(out_all)
    def emb_encode(qs):
        return _emb(qs)

    from src.agent import answer_question
    from src.embed_index import search
    import mlflow
    import src.generate as gen_mod

    latencies_clean = []  # clean Groq API time excl. backoff
    latencies_wall = []   # total wall including retries + retrieval
    cache_latencies = []  # wall for hits only
    relevance_scores = []
    cache_hits = 0
    fallback_routes = 0
    sims_all = []

    # helper for FAISS-only top-3 ids (no rerank)
    def faiss_only_ids(query, k=15, top3=3):
        import faiss
        meta_path = Path(cfg["paths"]["metadata_path"])
        index_path = Path(cfg["paths"]["index_path"])
        with open(meta_path, "rb") as f:
            chunks = pickle.load(f)
        idx = faiss.read_index(str(index_path))
        q_emb = emb_encode([query]).astype("float32")
        import faiss as _f
        _f.normalize_L2(q_emb)
        scores, ids = idx.search(q_emb, k)
        # ids[0] are faiss-ranked
        return [chunks[i]["id"] for i in ids[0][:top3] if i != -1]

    with mlflow.start_run(run_name=f"bench-{provider}"):
        for q in tqdm(queries, desc=f"bench {provider}"):
            t0_wall = time.time()
            result = answer_question(q, provider=provider, use_cache=True)
            wall = (time.time() - t0_wall) * 1000
            is_hit = bool(result.get("cache_hit"))
            if is_hit:
                cache_hits += 1
                clean = wall  # Redis lookup only
                cache_latencies.append(wall)
                total = wall
            else:
                # clean Groq latency from generate module (excl. backoff)
                clean = gen_mod.LAST_CLEAN_LATENCY_MS
                if clean is None:
                    clean = result.get("latency_ms", wall)
                total = wall

            latencies_clean.append(clean if clean is not None else wall)
            latencies_wall.append(total)

            if result.get("route") == "fallback":
                fallback_routes += 1

            hits = result.get("hits", [])
            if hits:
                q_emb = emb_encode([q])
                h_emb = emb_encode([hits[0]["text"]])
                sim = cosine_sim(q_emb[0], h_emb[0])
                relevance_scores.append(sim)
                sims_all.append(sim)
            else:
                relevance_scores.append(0)

            mlflow.log_metric("clean_gen_latency_ms", clean if clean else wall, step=len(latencies_clean))
            mlflow.log_metric("wall_latency_ms", wall, step=len(latencies_clean))
            if hits:
                mlflow.log_metric("relevance", relevance_scores[-1], step=len(latencies_clean))

        avg_clean = float(np.mean(latencies_clean)) if latencies_clean else 0
        avg_wall = float(np.mean(latencies_wall)) if latencies_wall else 0
        avg_cache = float(np.mean(cache_latencies)) if cache_latencies else 0
        avg_relevance = float(np.mean(relevance_scores)) if relevance_scores else 0
        relevance_pct = float(np.mean([1 if s > cfg["retrieval"]["similarity_threshold"] else 0 for s in relevance_scores]) * 100) if relevance_scores else 0
        hit_rate = float(cache_hits / len(queries) * 100) if queries else 0
        fallback_rate = float(fallback_routes / len(queries) * 100) if queries else 0

        # precision@3 for FAISS-only vs FAISS+rerank
        prec_faiss = []
        prec_rerank = []
        for q in queries:
            correct_ids = set(eval_labels.get(q, []))
            if not correct_ids:
                # no label -> skip (don't penalize fallback queries with no true chunk)
                continue
            # faiss-only
            f_ids = set(faiss_only_ids(q, k=cfg["retrieval"]["faiss_k"], top3=3))
            p_f = 1.0 if (f_ids & correct_ids) else 0.0
            prec_faiss.append(p_f)
            # reranked
            try:
                r_hits = search(q, k=cfg["retrieval"]["faiss_k"], rerank_k=3)
                r_ids = set([h["id"] for h in r_hits[:3]])
                p_r = 1.0 if (r_ids & correct_ids) else 0.0
                prec_rerank.append(p_r)
            except Exception as e:
                print(f"[warn] rerank prec failed for {q[:30]}: {e}")
                prec_rerank.append(0.0)

        prec_faiss_avg = float(np.mean(prec_faiss)*100) if prec_faiss else 0.0
        prec_rerank_avg = float(np.mean(prec_rerank)*100) if prec_rerank else 0.0
        prec_lift = prec_rerank_avg - prec_faiss_avg

        mlflow.log_metric("avg_clean_latency_ms", avg_clean)
        mlflow.log_metric("avg_wall_latency_ms", avg_wall)
        mlflow.log_metric("avg_cache_latency_ms", avg_cache)
        mlflow.log_metric("avg_relevance", avg_relevance)
        mlflow.log_metric("relevance_pct", relevance_pct)
        mlflow.log_metric("cache_hit_rate", hit_rate)
        mlflow.log_metric("fallback_rate", fallback_rate)
        mlflow.log_metric("precision_faiss_p3", prec_faiss_avg)
        mlflow.log_metric("precision_rerank_p3", prec_rerank_avg)
        mlflow.log_metric("precision_lift", prec_lift)
        mlflow.log_param("provider", provider)
        mlflow.log_param("device", device)

        print(f"\n[bench:{provider}] clean_gen={avg_clean:.0f}ms wall={avg_wall:.0f}ms cache_hit_lat={avg_cache:.0f}ms relevance={relevance_pct:.1f}% hit_rate={hit_rate:.1f}% fallback={fallback_rate:.1f}%")
        print(f"[bench:{provider}] precision@3 FAISS={prec_faiss_avg:.1f}% rerank={prec_rerank_avg:.1f}% lift={prec_lift:.1f}pp (n={len(prec_faiss)})")

        if wandb_run:
            import wandb
            wandb.log({
                "avg_clean_latency_ms": avg_clean,
                "avg_wall_latency_ms": avg_wall,
                "avg_cache_latency_ms": avg_cache,
                "avg_relevance": avg_relevance,
                "relevance_pct": relevance_pct,
                "cache_hit_rate": hit_rate,
                "fallback_rate": fallback_rate,
                "precision_faiss_p3": prec_faiss_avg,
                "precision_rerank_p3": prec_rerank_avg,
                "precision_lift": prec_lift,
                "provider": provider,
            })
            if sims_all:
                import wandb as wb
                table = wb.Table(data=[[s] for s in sims_all], columns=["similarity"])
                wandb.log({"similarity_hist": wb.plot.histogram(table, "similarity", title="retrieval similarity")})
            wandb.finish()

        summary = {
            "provider": provider,
            "num_queries": len(queries),
            "avg_clean_latency_ms": avg_clean,
            "avg_wall_latency_ms": avg_wall,
            "avg_cache_latency_ms": avg_cache,
            "avg_relevance": avg_relevance,
            "relevance_pct": relevance_pct,
            "cache_hit_rate": hit_rate,
            "fallback_rate": fallback_rate,
            "precision_faiss_p3": prec_faiss_avg,
            "precision_rerank_p3": prec_rerank_avg,
            "precision_lift_pp": prec_lift,
            "device": device,
        }
        out_path = Path(f"data/processed/bench_{provider}.json")
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[bench] saved {out_path}")
        return summary

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default="groq", choices=["groq", "ollama"])
    p.add_argument("--limit", type=int, default=25)
    args = p.parse_args()
    run_benchmark(provider=args.provider, limit=args.limit)
