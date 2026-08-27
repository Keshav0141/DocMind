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

# module-level model cache (perf fix #3)
_EMB_TOK = None
_EMB_MDL = None
_EMB_DEVICE = None
_CE_TOK = None
_CE_MDL = None
_CE_DEVICE = None

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except:
        return "cpu"

def _get_emb_model(device):
    global _EMB_TOK, _EMB_MDL, _EMB_DEVICE
    if _EMB_TOK is not None and _EMB_MDL is not None and _EMB_DEVICE == device:
        return _EMB_TOK, _EMB_MDL
    from transformers import AutoTokenizer, AutoModel
    import torch
    cfg = load_config()
    cache_path = Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    load_name = str(cache_path) if cache_path.exists() else cfg["models"]["embedding"]
    _EMB_TOK = AutoTokenizer.from_pretrained(load_name, local_files_only=cache_path.exists())
    _EMB_MDL = AutoModel.from_pretrained(load_name, local_files_only=cache_path.exists()).to(device)
    _EMB_MDL.eval()
    _EMB_DEVICE = device
    return _EMB_TOK, _EMB_MDL

def _get_ce_model(device):
    global _CE_TOK, _CE_MDL, _CE_DEVICE
    if _CE_TOK is not None and _CE_MDL is not None and _CE_DEVICE == device:
        return _CE_TOK, _CE_MDL
    from transformers import AutoTokenizer as CETok, AutoModelForSequenceClassification
    cfg = load_config()
    ce_name = cfg["models"]["reranker"]
    _CE_TOK = CETok.from_pretrained(ce_name)
    _CE_MDL = AutoModelForSequenceClassification.from_pretrained(ce_name).to(device)
    _CE_MDL.eval()
    _CE_DEVICE = device
    return _CE_TOK, _CE_MDL

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

    import torch
    tok, mdl = _get_emb_model(device)
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
    # ensure L2-normalized for IP == cosine (explicit, even though torch normalized)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    print(f"[embed] building FAISS index dim={dim} (IP, L2-normalized, k_recall={cfg['retrieval']['faiss_k']} -> rerank {cfg['retrieval']['rerank_k']})")
    index = faiss.IndexFlatIP(dim)  # cosine via normalized
    index.add(embeddings)
    # sanity: verify normalized
    norms = (embeddings**2).sum(axis=1)
    print(f"[embed] sanity norms mean={norms.mean():.4f} min={norms.min():.4f} max={norms.max():.4f} (should be ~1.0)")
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
    import faiss as _faiss
    _faiss.normalize_L2(q_emb)
    scores, ids = index.search(q_emb, 5)
    print(f"[test] query: {test_q}")
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0])):
        txt = chunks[idx]['text'][:120].encode('ascii', 'ignore').decode()
        src = chunks[idx]['source'][:60]
        print(f"  {rank+1}. score={score:.3f} src={src} text={txt}...")

    # quick rerank demo if cross-encoder available (cached)
    try:
        import torch
        # try cached
        try:
            ce_tok, ce_mdl = _get_ce_model(device)
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
    import torch
    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index(str(index_path))
    tok, mdl = _get_emb_model(device)
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
    faiss.normalize_L2(q_emb)
    # confirm k is specced 10-15
    print(f"[search] FAISS recall k={k} -> rerank_k={rerank_k} for query '{query[:40]}...'") if False else None
    scores, ids = index.search(q_emb, k)
    candidates = [(int(idx), float(score), chunks[idx]) for idx, score in zip(ids[0], scores[0]) if idx != -1]

    # rerank via cross-encoder (cached)
    try:
        ce_tok, ce_mdl = _get_ce_model(device)
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
