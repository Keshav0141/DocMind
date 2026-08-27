"""
embed_index.py — Day 1: embed chunks + FAISS + cross-encoder rerank
GPU-aware: uses cuda if torch.cuda.is_available()
"""
import json
import pickle
from pathlib import Path
import yaml
import numpy as np

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except:
        return "cpu"

def load_chunks(chunks_path: str):
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def build_index():
    cfg = load_config()
    device = get_device()
    print(f"[embed] device: {device}")
    chunks_path = Path(cfg["paths"]["chunks_path"])
    index_path = Path(cfg["paths"]["index_path"])
    meta_path = Path(cfg["paths"]["metadata_path"])

    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks not found at {chunks_path}. Run ingest.py first.")

    chunks = load_chunks(str(chunks_path))
    texts = [c["text"] for c in chunks]
    print(f"[embed] embedding {len(texts)} chunks with {cfg['models']['embedding']}")

    # Use transformers directly to avoid sentence-transformers segfault (see investigation)
    from transformers import AutoTokenizer, AutoModel
    import torch
    model_name = cfg["models"]["embedding"]
    # try local snapshot first, else hub
    cache_path = Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    load_name = str(cache_path) if cache_path.exists() else model_name
    tok = AutoTokenizer.from_pretrained(load_name, local_files_only=cache_path.exists())
    mdl = AutoModel.from_pretrained(load_name, local_files_only=cache_path.exists())
    mdl = mdl.to(device)
    mdl.eval()
    def _encode(texts_, batch_size=32):
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts_), batch_size):
                batch = texts_[i:i+batch_size]
                inputs = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out = mdl(**inputs)
                # mean pooling with attention mask
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                token_emb = out.last_hidden_state
                summed = (token_emb * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                emb = summed / counts
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs).astype("float32")
    embeddings = _encode(texts, batch_size=32)

    import faiss
    dim = embeddings.shape[1]
    print(f"[embed] building FAISS index dim={dim}")
    index = faiss.IndexFlatIP(dim)  # cosine via normalized
    index.add(embeddings)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"[embed] saved index to {index_path} ({index.ntotal} vectors)")

    # save metadata
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[embed] saved metadata to {meta_path}")

    # test query
    test_q = "What is LangChain?"
    # encode query with same method
    import torch
    def _encode_q(qs):
        from transformers import AutoTokenizer, AutoModel
        # reuse tok/mdl from above closure — recreate quickly
        return _encode(qs, batch_size=8)
    q_emb = _encode([test_q])
    scores, ids = index.search(q_emb, 5)
    print(f"[test] query: {test_q}")
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0])):
        txt = chunks[idx]['text'][:120].encode('ascii', 'ignore').decode()
        src = chunks[idx]['source'][:60]
        print(f"  {rank+1}. score={score:.3f} src={src} text={txt}...")

    # quick rerank demo if cross-encoder available (optional, uses transformers)
    try:
        from transformers import AutoTokenizer as CETok, AutoModelForSequenceClassification
        import torch
        ce_name = cfg["models"]["reranker"]
        # try cached, else download
        try:
            ce_tok = CETok.from_pretrained(ce_name)
            ce_mdl = AutoModelForSequenceClassification.from_pretrained(ce_name).to(device)
            ce_mdl.eval()
            top_k = min(10, len(texts))
            scores, ids = index.search(q_emb, top_k)
            pairs = [(test_q, chunks[i]["text"]) for i in ids[0]]
            ce_scores = []
            with torch.no_grad():
                for q_, d_ in pairs:
                    inputs = ce_tok(q_, d_, padding=True, truncation=True, max_length=512, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    out = ce_mdl(**inputs)
                    s = out.logits[0].item()
                    ce_scores.append(s)
            reranked = sorted(zip(ids[0], ce_scores), key=lambda x: x[1], reverse=True)[:3]
            print("[test] reranked top 3:")
            for idx, s in reranked:
                txt = chunks[idx]['text'][:120].encode('ascii','ignore').decode()
                print(f"  ce={s:.3f} text={txt}...")
        except Exception as inner:
            print(f"[warn] reranker model not available (download needed): {inner}")
    except Exception as e:
        print(f"[warn] reranker demo skipped: {e}")

    return index, chunks

def search(query: str, k: int = None, rerank_k: int = None):
    """Utility: embed query, FAISS search, cross-encoder rerank -> top rerank_k"""
    cfg = load_config()
    device = get_device()
    k = k or cfg["retrieval"]["faiss_k"]
    rerank_k = rerank_k or cfg["retrieval"]["rerank_k"]
    index_path = Path(cfg["paths"]["index_path"])
    meta_path = Path(cfg["paths"]["metadata_path"])
    if not index_path.exists():
        raise FileNotFoundError("FAISS index not found. Run embed_index.py first.")

    import faiss
    from transformers import AutoTokenizer, AutoModel
    import torch
    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index(str(index_path))
    # load embedding model via transformers (same as build_index)
    cache_path = Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    load_name = str(cache_path) if cache_path.exists() else cfg["models"]["embedding"]
    tok = AutoTokenizer.from_pretrained(load_name, local_files_only=cache_path.exists())
    mdl = AutoModel.from_pretrained(load_name, local_files_only=cache_path.exists())
    mdl = mdl.to(device)
    mdl.eval()
    def _encode_q(texts_, batch_size=8):
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts_), batch_size):
                batch = texts_[i:i+batch_size]
                inputs = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out = mdl(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                token_emb = out.last_hidden_state
                summed = (token_emb * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                emb = summed / counts
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs).astype("float32")
    q_emb = _encode_q([query])
    scores, ids = index.search(q_emb, k)
    candidates = [(int(idx), float(score), chunks[idx]) for idx, score in zip(ids[0], scores[0]) if idx != -1]

    # rerank via cross-encoder (optional)
    try:
        from transformers import AutoTokenizer as CETok, AutoModelForSequenceClassification
        ce_name = cfg["models"]["reranker"]
        ce_tok = CETok.from_pretrained(ce_name)
        ce_mdl = AutoModelForSequenceClassification.from_pretrained(ce_name).to(device)
        ce_mdl.eval()
        pairs = [(query, c["text"]) for _, _, c in candidates]
        ce_scores = []
        with torch.no_grad():
            for q_, d_ in pairs:
                inputs = ce_tok(q_, d_, padding=True, truncation=True, max_length=512, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out = ce_mdl(**inputs)
                s = out.logits[0].item()
                ce_scores.append(s)
        reranked = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)[:rerank_k]
        out = []
        for (idx, faiss_score, ch), ce_s in reranked:
            out.append({"id": ch["id"], "source": ch["source"], "text": ch["text"], "faiss_score": faiss_score, "rerank_score": float(ce_s)})
        return out
    except Exception as e:
        print(f"[warn] rerank failed, returning FAISS only: {e}")
        return [{"id": c["id"], "source": c["source"], "text": c["text"], "faiss_score": s, "rerank_score": s} for _, s, c in candidates[:rerank_k]]

if __name__ == "__main__":
    build_index()
