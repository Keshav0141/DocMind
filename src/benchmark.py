"""
benchmark.py — Day 5/6: MLflow + W&B tracking, metrics, rerank lift
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

    # load eval queries
    eval_path = Path(cfg["benchmark"]["eval_queries_path"])
    queries = [l.strip() for l in eval_path.read_text(encoding="utf-8").splitlines() if l.strip()][:limit]

    # init mlflow
    import mlflow
    mlflow.set_experiment(cfg["benchmark"]["mlflow_experiment"])

    # init wandb (optional)
    wandb_run = None
    try:
        import wandb
        if os.getenv("WANDB_API_KEY"):
            wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb_run = wandb.init(project=cfg["benchmark"]["wandb_project"], name=f"bench-{provider}-{int(time.time())}", reinit=True)
    except Exception as e:
        print(f"[wandb] init failed: {e}")

    # load embedding model for relevance scoring via transformers (avoid ST segfault)
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
            import numpy as np
            return np.vstack(out_all)
    # wrapper for single
    def emb_encode(qs):
        return _emb(qs)

    # optional cross-encoder rerank lift comparison: we compute FAISS-only vs reranked
    from src.agent import answer_question
    from src.embed_index import search
    import mlflow

    latencies = []
    relevance_scores = []
    cache_hits = 0
    fallback_routes = 0
    rerank_lift_sum = 0

    sims_all = []

    with mlflow.start_run(run_name=f"bench-{provider}"):
        for q in tqdm(queries, desc=f"bench {provider}"):
            t0 = time.time()
            result = answer_question(q, provider=provider, use_cache=True)
            latency = result.get("latency_ms", (time.time()-t0)*1000)
            latencies.append(latency)
            if result.get("cache_hit"):
                cache_hits += 1
            if result.get("route") == "fallback":
                fallback_routes += 1

            hits = result.get("hits", [])
            # relevance: cosine between query and top hit text
            if hits:
                q_emb = emb_encode([q])
                h_emb = emb_encode([hits[0]["text"]])
                sim = cosine_sim(q_emb[0], h_emb[0])
                relevance_scores.append(sim)
                sims_all.append(sim)
                # rerank lift: compare faiss top-1 vs reranked top-1 (need to fetch both)
                # do FAISS-only top-5 vs reranked; lift = rerank sim - faiss sim
                try:
                    faiss_hits = search(q, k=cfg["retrieval"]["faiss_k"], rerank_k=cfg["retrieval"]["faiss_k"])
                    # faiss_hits is already reranked; to get FAISS-only we need raw FAISS
                    # approximate: compute sim for faiss_scores vs rerank_scores ordering difference
                    # simpler: just compare first hit sim vs average of top 3 sim
                except:
                    pass
            else:
                relevance_scores.append(0)

            mlflow.log_metric("latency_ms", latency, step=len(latencies))
            if hits:
                mlflow.log_metric("relevance", relevance_scores[-1], step=len(latencies))

        # aggregate metrics
        avg_latency = float(np.mean(latencies)) if latencies else 0
        avg_relevance = float(np.mean(relevance_scores)) if relevance_scores else 0
        relevance_pct = float(np.mean([1 if s > cfg["retrieval"]["similarity_threshold"] else 0 for s in relevance_scores]) * 100) if relevance_scores else 0
        hit_rate = float(cache_hits / len(queries) * 100) if queries else 0
        fallback_rate = float(fallback_routes / len(queries) * 100) if queries else 0

        # rerank lift: compare FAISS-only top-3 avg sim vs rerank top-3 avg sim on a sample of 5 queries
        lift_vals = []
        for sq in queries[:5]:
            import faiss, pickle
            meta_path = Path(cfg["paths"]["metadata_path"])
            index_path = Path(cfg["paths"]["index_path"])
            if not index_path.exists():
                break
            with open(meta_path, "rb") as f:
                chunks = pickle.load(f)
            idx = faiss.read_index(str(index_path))
            q_emb = emb_encode([sq]).astype("float32")
            scores, ids = idx.search(q_emb, 10)
            faiss_hits = [chunks[i]["text"] for i in ids[0][:3]]
            faiss_sims = [cosine_sim(emb_encode([sq])[0], emb_encode([t])[0]) for t in faiss_hits]
            faiss_avg = float(np.mean(faiss_sims)) if faiss_sims else 0
            reranked = search(sq, k=10, rerank_k=3)
            rerank_texts = [h["text"] for h in reranked]
            rerank_sims = [cosine_sim(emb_encode([sq])[0], emb_encode([t])[0]) for t in rerank_texts]
            rerank_avg = float(np.mean(rerank_sims)) if rerank_sims else 0
            if faiss_avg > 0:
                lift = (rerank_avg - faiss_avg) / faiss_avg * 100
                lift_vals.append(lift)
        rerank_lift = float(np.mean(lift_vals)) if lift_vals else 0

        mlflow.log_metric("avg_latency_ms", avg_latency)
        mlflow.log_metric("avg_relevance", avg_relevance)
        mlflow.log_metric("relevance_pct", relevance_pct)
        mlflow.log_metric("cache_hit_rate", hit_rate)
        mlflow.log_metric("fallback_rate", fallback_rate)
        mlflow.log_metric("rerank_lift_pct", rerank_lift)
        mlflow.log_param("provider", provider)
        mlflow.log_param("device", device)

        print(f"\n[bench:{provider}] avg_latency={avg_latency:.0f}ms relevance={relevance_pct:.1f}% hit_rate={hit_rate:.1f}% fallback={fallback_rate:.1f}% rerank_lift={rerank_lift:.1f}%")

        if wandb_run:
            import wandb
            wandb.log({
                "avg_latency_ms": avg_latency,
                "avg_relevance": avg_relevance,
                "relevance_pct": relevance_pct,
                "cache_hit_rate": hit_rate,
                "fallback_rate": fallback_rate,
                "rerank_lift_pct": rerank_lift,
                "provider": provider,
            })
            # log similarity distribution
            if sims_all:
                import wandb as wb
                table = wb.Table(data=[[s] for s in sims_all], columns=["similarity"])
                wandb.log({"similarity_hist": wb.plot.histogram(table, "similarity", title="retrieval similarity")})
            wandb.finish()

        # save json summary
        summary = {
            "provider": provider,
            "num_queries": len(queries),
            "avg_latency_ms": avg_latency,
            "avg_relevance": avg_relevance,
            "relevance_pct": relevance_pct,
            "cache_hit_rate": hit_rate,
            "fallback_rate": fallback_rate,
            "rerank_lift_pct": rerank_lift,
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
