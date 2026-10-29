import hashlib
import json
import os
import time

import redis
import requests
from fastapi import FastAPI, Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()

redis_client = redis.Redis(
    host=os.environ.get("REDIS_HOST", "redis-service"),
    port=6379,
    decode_responses=True,
)

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")
CACHE_TTL_SECONDS = 3600

# Prometheus metrics: these accumulate in memory and get exposed via /metrics
cache_hits_total = Counter("cache_hits_total", "Total number of cache hits")
cache_misses_total = Counter("cache_misses_total", "Total number of cache misses")
request_latency_seconds = Histogram("request_latency_seconds", "Latency of chat completion requests")


class ChatRequest(BaseModel):
    model: str
    messages: list


def make_cache_key(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True)
    return "llm_cache:" + hashlib.sha256(canonical.encode()).hexdigest()


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    start_time = time.time()
    payload = request.dict()
    cache_key = make_cache_key(payload)

    cached = redis_client.get(cache_key)
    if cached:
        cache_hits_total.inc()
        response = json.loads(cached)
        response["_cache_hit"] = True
        request_latency_seconds.observe(time.time() - start_time)
        return response

    cache_misses_total.inc()
    vllm_response = requests.post(VLLM_URL, json=payload, timeout=120)
    vllm_response.raise_for_status()
    result = vllm_response.json()

    redis_client.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(result))
    result["_cache_hit"] = False
    request_latency_seconds.observe(time.time() - start_time)
    return result


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
