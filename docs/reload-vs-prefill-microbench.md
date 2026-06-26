# Reload (CPU→GPU) vs Prefill — End-to-End Microbenchmark

**Date:** 2026-06-27
**Setup:** Qwen3-4B (bf16), single RTX 3090 (GPU 3), vLLM 0.10.2 editable,
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, `--no-enable-prefix-caching`,
`--max-num-batched-tokens 65536` (so a long prompt prefills in one step).
KV geometry: 36 layers × 8 KV heads × 128 head_dim × 2 (K&V) × bf16 = **144 KB/token**.

## Question

When an agent step returns and its KV cache has been evicted from GPU, there
are two ways to resume:
- **prefill** — recompute the entire prefix KV from scratch
- **reload** — load the prefix KV back from CPU memory via MimirConnector

Which is faster **end-to-end**, including ALL connector per-step overhead?
This is the load-bearing question for the "TTL-expiry → drop to CPU → reload"
optimization (innovation point #2).

## Protocol

Both paths run with vLLM's own GPU prefix caching **disabled**, so the only KV
available to reload is what MimirConnector stored in CPU; prefill always
recomputes. We measure single-step TTFT (request submit → first token) for
prefix lengths 4K / 16K / 32K. Each (length, path) = 5 timed samples after 2
warmups; reload mode uses a **fresh server process per length** to avoid
cross-length store pollution (a long prompt shares the shorter prompt's prefix
and would otherwise hit the stale short prefix instead of storing its own).

## Results

| Prefix | prefill (s) | reload (s) | speedup | theoretical PCIe (s) | overhead share |
|--------|-------------|------------|---------|----------------------|----------------|
| 4K  (4080 tok)  | 0.538 | 0.175 | 3.1× | 0.029 | ~83% |
| 16K (16368 tok) | 3.041 | 0.646 | 4.7× | 0.117 | ~82% |
| 32K (32752 tok) | 8.871 | 1.191 | 7.4× | 0.235 | ~80% |

(medians; speedup = prefill/reload; "overhead share" = 1 − theoretical_PCIe/reload)

## Honest conclusions

1. **Reload wins end-to-end at every length, and the margin grows with prefix
   length** (3.1× → 4.7× → 7.4×). The "TTL-expiry → drop to CPU → reload"
   path is **vindicated as a real net win**, not just a theoretical one. This
   is the foundation for innovation point #2.

2. **But the win is far below the physical ceiling.** Raw PCIe transfer is
   10–25× faster than prefill; the realized speedup is only 3–7×. The
   connector's per-step overhead consumes ~80% of the reload time at every
   length. The dominant cost is NOT the H2D copy — it is the synchronous,
   per-layer, per-request `.to(cpu)`/`.to(cuda)` with no pipelining.

3. **The overhead is roughly constant-share, not growing with length** (~80%
   across 4K–32K). This means it scales with the number of layers/tokens moved
   (each layer copied separately), not a fixed per-request tax. Layer-wise
   async pipelining (overlap layer N's copy with layer N+1's compute, as
   LMCache does) is the clear lever to recover the missing margin.

4. **No silent fallback.** After the key-consistency fix, every reload sample
   logged a clean cache hit at the full prefix length with zero load-misses —
   the numbers above are genuine reloads, not prefill masquerading as reload.

## What this means for the project

- **Innovation point #2 (drop-to-CPU reload) is feasible and worth doing.**
  Even an unoptimized connector already beats prefill by 3–7× end-to-end.
- **The connector must be optimized** (layer-wise pipelining) to approach the
  10–25× physical ceiling. This is a concrete, measurable next step with a
  clear target: close the ~80% overhead gap.
- The "equilibrator" idea (dynamically choose reload vs prefill per request)
  is supported by this data: reload's advantage grows with length, so a
  length-aware policy would pick reload for long prefixes and could fall back
  to prefill for very short ones where fixed overhead dominates.

## Reproducing

```bash
# reload mode (fresh process per length)
CUDA_VISIBLE_DEVICES=3 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
vllm serve /data/models/Qwen3-4B --port 8199 \
  --kv-transfer-config '{"kv_connector":"MimirConnector","kv_role":"kv_both"}' \
  --no-enable-prefix-caching --max-model-len 40960 \
  --max-num-batched-tokens 65536 --gpu-memory-utilization 0.85 --dtype bfloat16
python benchmarks/mimir/reload_vs_prefill.py --mode reload --length 32768 --n 5

# prefill mode (no connector)
CUDA_VISIBLE_DEVICES=3 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
vllm serve /data/models/Qwen3-4B --port 8199 \
  --no-enable-prefix-caching --max-model-len 40960 \
  --max-num-batched-tokens 65536 --gpu-memory-utilization 0.85 --dtype bfloat16
python benchmarks/mimir/reload_vs_prefill.py --mode prefill --length 32768 --n 5
```

## Caveats

- Single-request, no concurrency. Under real multi-agent contention the
  reload path's GPU memory pressure and scheduling interaction differ; this
  microbenchmark isolates the pure reload-vs-prefill cost, not the full
  serving story (that is the benchmark harness's job, per the benchmark-design
  rules).
- `max-num-batched-tokens 65536` forces single-step prefill so the connector
  stores the whole prefix at once. Real chunked-prefill serving needs the
  connector to accumulate a prefix across chunks — a known TODO.
- The connector runs in single-process mode (`VLLM_ENABLE_V1_MULTIPROCESSING=0`)
  so scheduler/worker share the CPU store. Multi-worker deployment needs
  cross-process sharing.
