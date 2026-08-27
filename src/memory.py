"""
memory.py — Day 4: Redis cache helpers
"""
import os
import json
import hashlib
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv()
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def _client():
    url = os.getenv("REDIS_URL")
    if not url:
        raise ValueError("REDIS_URL not set")
    import redis
    return redis.from_url(url, decode_responses=True, socket_connect_timeout=5, socket_timeout=5)

def query_hash(query: str) -> str:
    cfg = load_config()
    prefix = cfg["redis"]["key_prefix"]
    h = hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]
    return f"{prefix}{h}"

def get_cached(query: str):
    key = query_hash(query)
    try:
        r = _client()
        val = r.get(key)
        if val:
            return json.loads(val)
        return None
    except Exception as e:
        print(f"[redis] get failed: {e}")
        return None

def set_cached(query: str, response: dict):
    cfg = load_config()
    key = query_hash(query)
    ttl = cfg["redis"]["ttl_seconds"]
    try:
        r = _client()
        r.set(key, json.dumps(response, ensure_ascii=False), ex=ttl)
        return True
    except Exception as e:
        print(f"[redis] set failed: {e}")
        return False

def cache_stats():
    """Count keys with prefix (for benchmark reporting)"""
    try:
        r = _client()
        cfg = load_config()
        prefix = cfg["redis"]["key_prefix"]
        # scan
        count = 0
        for _ in r.scan_iter(match=f"{prefix}*", count=100):
            count += 1
        return count
    except Exception as e:
        print(f"[redis] stats failed: {e}")
        return 0

if __name__ == "__main__":
    # quick test
    print("hash:", query_hash("What is LangChain?"))
    ok = set_cached("test query", {"answer": "hello", "hits": 1})
    print("set:", ok, "get:", get_cached("test query"))
    print("keys:", cache_stats())
