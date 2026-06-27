#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Trace-replay serving benchmark: native vLLM vs Mimir2.

Replays collected agent traces against a vLLM server to measure end-to-end
serving performance under realistic multi-agent load. This is the benchmark
that any KV-management optimization must improve end-to-end (per the
project's "end-to-end is the only truth" rule).

Design (mirrors Continuum's evaluation, avoids the v1 benchmark's fatal flaws)
--------------------------------------------------------------------------
- ASYNC agents: N agent programs each advance independently through their
  trace turns. They never synchronize ("barrier") with each other.
- POISSON arrival: agent programs are injected into the system according to a
  Poisson process (not all at once).
- NEVER drain: agents contend for GPU KV continuously; we never flush the
  cache between turns (v1's _drain_all was the root failure — it destroyed
  the cross-turn KV that pin/SSC are meant to preserve, and hid real
  contention).
- ACCUMULATING prefix: each agent's turn prompt grows turn over turn (the
  prefix-reuse case that pin/SSC optimize for). We replay the recorded
  per-turn prompt-token counts, scaled by a factor so the context reaches a
  length where reload advantage is measurable.
- METRICS: per-agent job completion time (primary), per-request P50/P90 TTFT.
  Technical metrics (hit rate) are auxiliary, never a substitute.

Payload
-------
Traces record token *counts* per turn, not DeepSeek's actual text (which is
DeepSeek-specific and irrelevant to Qwen3-4B serving). We synthesize a filler
prompt of the recorded length per turn, growing across turns to emulate the
accumulating context. The content does not matter for KV-management cost;
only the token budget and its growth do.

Usage
-----
  # start a server (native or Mimir2 mode), then:
  python benchmarks/mimir/replay_benchmark.py \
    --traces benchmarks/mimir/traces/batch30 \
    --num-agents 8 --arrival-rate 0.5 --scale 4 --port 8199 \
    --label native --out benchmarks/mimir/results/native.json
"""
import argparse
import asyncio
import json
import os
import random
import statistics
import time
from pathlib import Path

import httpx

FILLERS = [
    "In the study of large language model inference, KV cache management is "
    "the central problem being addressed by this work. ",
    "Agents that interleave reasoning and tool calls accumulate long contexts "
    "across turns, making memory reuse a key efficiency concern. ",
    "Resource-constrained serving requires careful eviction and reload of "
    "attention state to keep latency bounded under contention. ",
    "The paged attention design divides the key-value buffer into fixed-size "
    "blocks that can be allocated and freed on demand per request. ",
    "Multi-turn workloads differ from single-shot generation because their "
    "prefixes are reused across steps separated by tool-call gaps. ",
]


def _filler_for(agent_id: int) -> str:
    """Pick a filler text per agent so different agents do NOT share a prefix.

    If every agent used the same filler, vLLM's prefix cache would trivially
    hit across agents (they share the identical text prefix), producing a fake
    ~99% hit rate that is not real cross-turn reuse. Per-agent distinct
    fillers keep prefix-cache hits honest: only the same agent's own growing
    prefix across its turns should hit.
    """
    return FILLERS[agent_id % len(FILLERS)]


def tokens_to_text(n_tokens: int, agent_id: int = 0) -> str:
    """Roughly n_tokens of neutral filler (~22 tokens/sentence for Qwen3),
    using a per-agent filler so distinct agents don't share a prefix."""
    filler = _filler_for(agent_id)
    n = max(1, n_tokens // 22)
    return filler * n


class Trace:
    """One agent program's turn schedule, loaded from a collected trace.

    The trace supplies the REAL multi-turn decision structure (turn count,
    per-turn prompt-token counts and their growth) from DeepSeek running a
    BFCL task. Two dims are tuned at replay time, not stored in the trace:
      - prompt length is scaled (so context reaches a length where reload
        advantage is measurable);
      - tool latency is sampled from a fitted distribution (BFCL's reported
        mean/std), since our tools are synthesized and have no real latency.
    Decode length per turn IS taken from the trace (reasoning + completion
    tokens combined): from the engine's view every decoded token is one
    forward pass, thinking or not, so the total is the realistic load.
    """

    def __init__(self, path: str, scale: float, rng: random.Random):
        d = json.loads(Path(path).read_text())
        self.task_id = d["task_id"]
        self.turns = []
        cum = 0
        for t in d["turns"]:
            # Scale prompt tokens to reach a length where reload advantage is
            # measurable; clamp to a sane max to fit 24GB.
            scaled = min(int(t["prompt_tokens"] * scale), 30000)
            cum = max(cum, scaled)  # context only grows across turns
            # Real decode length: reasoning + completion combined.
            decode = (t.get("reasoning_tokens", 0) + t.get("completion_tokens", 0))
            # Drop the reasoning portion counted inside completion_tokens if
            # the trace stored it separately (DeepSeek double-counts). The
            # trace stores completion_tokens as total incl reasoning, plus
            # reasoning_tokens separately; use completion_tokens as-is.
            decode = t.get("completion_tokens", 1) or 1
            self.turns.append({
                "prompt_tokens": cum,
                "decode_tokens": max(1, decode),
                "has_tool_call": t.get("tool_name") is not None,
            })

    def __len__(self):
        return len(self.turns)


def sample_tool_latency(rng: random.Random, mean: float = 1.9,
                        std: float = 2.1) -> float:
    """Sample a tool-call latency (seconds) from a lognormal fitted to BFCL's
    reported web-search distribution (mean ~1.9s, std ~2.1s). Lognormal gives
    the right long-tail shape. Tunable: if a different source fits our needs
    better we can swap the params, but BFCL is the named baseline."""
    import math
    # lognormal: given target mean m and std s, derive mu/sigma.
    var = std * std
    mu = math.log(mean * mean / math.sqrt(var + mean * mean))
    sigma = math.sqrt(math.log(1 + var / (mean * mean)))
    return max(0.05, rng.lognormvariate(mu, sigma))


async def run_agent(client: httpx.AsyncClient, base_url: str, model: str,
                    trace: Trace, agent_id: int, rng: random.Random,
                    ttfts: list) -> float:
    """Advance one agent through its trace turns asynchronously.

    Each turn: prefill a growing prompt, decode the trace's real token count,
    then (if the turn had a tool call) sleep a sampled tool latency — the gap
    during which the agent's KV sits idle in GPU and pin/TTL/offload decisions
    matter. Returns job completion time; records per-turn TTFT.
    """
    start = time.perf_counter()
    n_turns = len(trace.turns)
    for turn_idx, turn in enumerate(trace.turns):
        prompt = tokens_to_text(turn["prompt_tokens"], agent_id)
        is_last_step = 1 if turn_idx == n_turns - 1 else 0
        has_tool = 1 if turn["has_tool_call"] else 0
        # Sample this turn's tool latency now, so we can pass it to the
        # scheduler (it needs the duration to decide pinning BEFORE the gap
        # starts) and then sleep for it after the request completes.
        tool_dur = sample_tool_latency(rng) if turn["has_tool_call"] else 0.0
        body = {
            "model": model,
            "prompt": prompt,
            "max_tokens": turn["decode_tokens"],  # real decode length
            "temperature": 0,
            # Mimir agent metadata: job_id groups one agent's turns so the
            # scheduler can pin KV across the tool gap; is_last_step suppresses
            # pinning after the final turn; has_tool_call + tool_duration let
            # the scheduler decide whether to pin (Continuum's set_up_pin).
            # Top-level vllm_xargs (not extra_body) because vLLM reads it from
            # the request body, and vllm_xargs forbids bool so use 0/1.
            "vllm_xargs": {
                "job_id": f"agent_{agent_id}",
                "is_last_step": is_last_step,
                "has_tool_call": has_tool,
                "tool_duration": round(tool_dur, 3),
            },
        }
        t0 = time.perf_counter()
        r = await client.post(f"{base_url}/v1/completions", json=body,
                              timeout=600)
        r.raise_for_status()
        ttft = time.perf_counter() - t0
        ttfts.append(ttft)
        # Tool-call gap: only on turns that actually called a tool (the last
        # turn gives the final answer and has no tool call). Sleep the
        # duration we already sampled and passed to the scheduler.
        if turn["has_tool_call"]:
            await asyncio.sleep(tool_dur)
    return time.perf_counter() - start


async def main_async(args):
    rng = random.Random(args.seed)
    trace_files = sorted(Path(args.traces).glob("web_search_*.json"))
    if not trace_files:
        raise SystemExit(f"no traces in {args.traces}")
    traces = [Trace(str(p), args.scale, rng) for p in trace_files]
    print(f"loaded {len(traces)} traces; scaling prompt tokens by {args.scale}")

    base_url = f"http://localhost:{args.port}"
    model = args.model

    # Poisson arrival: schedule agent start times.
    arrivals = []
    t = 0.0
    for i in range(args.num_agents):
        t += rng.expovariate(args.arrival_rate)
        arrivals.append(t)

    timings = []  # per-agent job completion times
    ttfts = []    # per-turn TTFT across all agents

    async with httpx.AsyncClient() as client:
        # Wait for server readiness
        for _ in range(60):
            try:
                r = await client.get(f"{base_url}/v1/models", timeout=5)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(2)

        wall_start = time.perf_counter()
        tasks = []
        for i, start_offset in enumerate(arrivals):
            trace = traces[i % len(traces)]
            # Schedule this agent to start at wall_start + start_offset.
            delay = start_offset
            tasks.append(_delayed_agent(client, base_url, model, trace, i,
                                        delay, rng, timings, ttfts))
        await asyncio.gather(*tasks)
        wall_total = time.perf_counter() - wall_start

    # Pressure evidence: scrape the server log for GPU KV cache usage peaks and
    # preemption. Without this, a "fast" result may just mean no real pressure
    # (v1's failure mode). We refuse to trust results where pressure is absent.
    pressure = _scrape_pressure(args.server_log)

    # Aggregate
    result = {
        "label": args.label,
        "num_agents": args.num_agents,
        "arrival_rate": args.arrival_rate,
        "scale": args.scale,
        "wall_total_s": round(wall_total, 2),
        "job_completion_s": {
            "median": round(statistics.median(timings), 3),
            "mean": round(statistics.mean(timings), 3),
            "p90": round(_percentile(timings, 90), 3),
            "max": round(max(timings), 3),
            "samples": [round(x, 3) for x in timings],
        },
        "ttft_s": {
            "median": round(statistics.median(ttfts), 3),
            "mean": round(statistics.mean(ttfts), 3),
            "p50": round(_percentile(ttfts, 50), 3),
            "p90": round(_percentile(ttfts, 90), 3),
            "n": len(ttfts),
        },
        "pressure_evidence": pressure,
    }
    print(json.dumps(result, indent=2))
    # Explicit warning if pressure looks absent (guard against v1-style false
    # fast results).
    if pressure["gpu_kv_usage_max"] < 0.5:
        print("\nWARNING: GPU KV usage peaked below 50% — little real memory "
              "pressure; Mimir2 has no room to win. Tune num_agents/scale up.")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")


async def _delayed_agent(client, base_url, model, trace, agent_id, delay,
                         rng, timings, ttfts):
    await asyncio.sleep(delay)
    try:
        jct = await run_agent(client, base_url, model, trace, agent_id, rng,
                              ttfts)
        timings.append(jct)
    except Exception as e:
        print(f"agent {agent_id} failed: {e}")
        timings.append(float("nan"))


def _percentile(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = int(round((len(s) - 1) * p / 100))
    return s[k]


def _scrape_pressure(server_log: str) -> dict:
    """Scrape the vLLM server log for GPU KV cache usage peaks and preemption
    counts. vLLM prints a periodic stats line like:
      'GPU KV cache usage: 42.3%, Prefix cache hit rate: 88.1%'
    and logs 'Preempted' when requests are preempted. This is our honest
    pressure evidence — without it a fast result may just mean no contention.
    """
    import re
    usage_vals = []
    hit_vals = []
    preempted = 0
    if server_log and Path(server_log).exists():
        text = Path(server_log).read_text(errors="ignore")
        for m in re.finditer(r"GPU KV cache usage:\s*([\d.]+)%", text):
            usage_vals.append(float(m.group(1)))
        for m in re.finditer(r"Prefix cache hit rate:\s*([\d.]+)%", text):
            hit_vals.append(float(m.group(1)))
        preempted = text.count("Preempted:")
    return {
        "gpu_kv_usage_max": round(max(usage_vals), 1) if usage_vals else 0.0,
        "gpu_kv_usage_mean": round(statistics.mean(usage_vals), 1) if usage_vals else 0.0,
        "prefix_hit_rate_last": round(hit_vals[-1], 1) if hit_vals else 0.0,
        "preempted_count": preempted,
        "samples": len(usage_vals),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--num-agents", type=int, default=8)
    ap.add_argument("--arrival-rate", type=float, default=0.5,
                    help="Poisson rate (agents/sec)")
    ap.add_argument("--scale", type=float, default=4.0,
                    help="multiply recorded prompt tokens by this to reach "
                         "a length where reload advantage is measurable")
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--model", default="/data/models/Qwen3-4B")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--server-log", default=None,
                    help="path to vLLM server log, to scrape pressure evidence")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
