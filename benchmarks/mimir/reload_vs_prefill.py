#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end microbenchmark: reload (CPU->GPU via MimirConnector) vs prefill.

Question this answers
---------------------
When an agent step returns and its KV cache has been evicted from the GPU,
there are two ways to resume:
  A. prefill   - recompute the entire prefix KV from scratch
  B. reload    - load the prefix KV back from CPU memory via MimirConnector

Which is faster end-to-end, and by how much, including ALL connector per-step
overhead (get_num_new_matched_tokens, build_connector_meta, the actual H2D
copy)? This is the load-bearing question for the "TTL-expiry -> drop to CPU ->
reload" optimization.

Protocol
--------
Both paths run with vLLM's own GPU prefix caching DISABLED, so the only KV
available to path B is what MimirConnector stored in CPU; path A always
recomputes. We measure single-step TTFT (request submit -> first token) for
prefix lengths 4K / 16K / 32K.

For each (length, path) we take N samples and report median + min/max. A fresh
server process is used per length to keep state clean.

How to run
----------
1. Start a server (one of the two modes) on GPU 3:
     # reload mode (connector on, prefix cache off)
     CUDA_VISIBLE_DEVICES=3 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
     vllm serve /data/models/Qwen3-4B --port 8199 \
       --kv-transfer-config '{"kv_connector":"MimirConnector","kv_role":"kv_both"}' \
       --no-enable-prefix-caching --max-model-len 40960 \
       --gpu-memory-utilization 0.85 --dtype bfloat16
     # prefill mode: same but WITHOUT the --kv-transfer-config line
2. python benchmarks/mimir/reload_vs_prefill.py --mode {reload|prefill} --length 4096 --n 5
3. Collect outputs into a table for both modes/lengths and compare medians.
"""
import argparse
import json
import statistics
import time

import requests

URL = "http://localhost:8199/v1/completions"
MODEL = "/data/models/Qwen3-4B"
# A neutral filler; repeated to hit the target prefix length. We measure a
# fixed prefix, so the content does not matter for the KV-transfer cost.
FILLER = "In the study of large language model inference, KV cache management " \
         "is the central problem being addressed by this work. "


def build_prompt(target_tokens: int) -> str:
    # Calibrated ~22 tokens per filler sentence for Qwen3-4B tokenizer.
    # We just need to land near the target so KV size scales across 4K/16K/32K;
    # exactness is not required, only that the three lengths differ materially.
    n = max(1, target_tokens // 22)
    return FILLER * n


def measure_ttft(prompt: str, n: int, warmup: int = 1) -> list[float]:
    """Return list of TTFT samples (seconds) for n requests after warmup.

    For 'reload' mode: the first request stores the prefix; subsequent requests
    reload it. warmup>=1 ensures the store is populated before timing.
    For 'prefill' mode: every request recomputes (connector off); warmup still
    runs once to stabilize cudagraph etc.
    """
    headers = {"Content-Type": "application/json"}
    body = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 1,  # we only care about TTFT
        "temperature": 0,
    }
    samples = []
    for i in range(warmup + n):
        t0 = time.perf_counter()
        r = requests.post(URL, headers=headers, json=body, timeout=600)
        dt = time.perf_counter() - t0
        r.raise_for_status()
        if i >= warmup:
            samples.append(dt)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["reload", "prefill"])
    ap.add_argument("--length", type=int, required=True,
                    help="approx target prefix tokens, e.g. 4096/16384/32768")
    ap.add_argument("--n", type=int, default=5, help="timed samples")
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    prompt = build_prompt(args.length)
    samples = measure_ttft(prompt, args.n, warmup=args.warmup)

    result = {
        "mode": args.mode,
        "target_length": args.length,
        "n": len(samples),
        "median_s": round(statistics.median(samples), 4),
        "mean_s": round(statistics.mean(samples), 4),
        "min_s": round(min(samples), 4),
        "max_s": round(max(samples), 4),
        "samples_s": [round(s, 4) for s in samples],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
