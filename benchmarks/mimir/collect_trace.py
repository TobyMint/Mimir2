#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect agent traces by running BFCL v4 web-search tasks on DeepSeek-v4-pro.

Purpose
-------
We need *realistic* agent load traces to drive the serving benchmark: real
multi-turn ReAct behavior (how many turns, how the context grows, which tools
get called, what tool-call latencies look like). DeepSeek-v4-pro plays the
agent brain; its decisions are real. The tool *results* are synthesized text
(controlled length) because we don't have a real search backend, and the tool
*latencies* are sampled from a fitted distribution — this is the deliberate
"trace + tuning" arrangement: real decisions, tunable latency/result dims.

What gets recorded per task (one JSONL line, one trace file per task)
--------------------------------------------------------------------
- task_id, question
- turns: list of per-turn records:
    - prompt_tokens, completion_tokens (from DeepSeek usage)
    - tool_name, tool_args
    - tool_latency_s (sampled)
    - tool_result_tokens (length of synthesized result)
    - assistant_text (the thought/text DeepSeek emitted, if any)
- total_turns, final_answer_tokens

The serving benchmark replays these traces against vLLM (Qwen3-4B): each turn
becomes an LLM request whose prompt is the accumulating context, with the
recorded tool latency as the gap before the next request.

Usage
-----
  export DEEPSEEK_API_KEY=...
  python benchmarks/mimir/collect_trace.py --num-tasks 3 --out benchmarks/mimir/traces/sample
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import requests

API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-pro"
MAX_TURNS = 8  # safety cap; we steer the agent to converge within ~6 turns

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return a short summary of top "
                           "results for the query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "the search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_page",
            "description": "Fetch and return the text content of a web page "
                           "URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "the page URL"}
                },
                "required": ["url"],
            },
        },
    },
]

# Tool-latency distribution (seconds), lognormal-ish via sampling from a
# fixed pool with a long tail, mirroring Continuum's observation that most
# tool calls are short but a few are very long. Tunable.
TOOL_LATENCY_POOL = [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
TOOL_LATENCY_WEIGHTS = [30, 25, 15, 10, 6, 4, 3, 2, 2, 1, 1, 1]


def sample_latency(rng: random.Random) -> float:
    return rng.choices(TOOL_LATENCY_POOL, TOOL_LATENCY_WEIGHTS, k=1)[0]


# Synthesized tool results. We keep them generic but plausible and control
# token length via repetition. The content matters less than the token budget
# for the serving benchmark.
SEARCH_RESULT_TEMPLATE = (
    "Search results for '{query}':\n"
    "1. {query} - Wikipedia, the free encyclopedia. This article covers the "
    "topic in depth, including history, key figures, and recent developments. "
    "The subject is notable for several reasons discussed across multiple "
    "sections.\n"
    "2. {query}: an overview. A reputable source summarizes the main points, "
    "noting that the answer depends on the specific timeframe and definition "
    "used.\n"
    "3. News report on {query}. Recent coverage indicates evolving details "
    "and expert commentary from multiple perspectives.\n"
)
PAGE_RESULT_TEMPLATE = (
    "Page content for {url}:\n"
    "This page discusses the requested topic in detail. Key points: the "
    "subject has several relevant attributes; historical context dates back "
    "several decades; and recent updates provide additional nuance. Multiple "
    "paragraphs elaborate on related subtopics, supported by references and "
    "data tables that contextualize the main claims made throughout the "
    "article. Readers should note that some figures are provisional.\n"
)


def synthesize_result(name: str, args: dict, rng: random.Random) -> str:
    """Return a plausible, length-controlled tool result string."""
    if name == "web_search":
        base = SEARCH_RESULT_TEMPLATE.format(query=args.get("query", "topic"))
        # vary length a bit so contexts aren't all identical
        reps = rng.randint(1, 3)
        return base * reps
    elif name == "retrieve_page":
        base = PAGE_RESULT_TEMPLATE.format(url=args.get("url", "page"))
        reps = rng.randint(1, 4)
        return base * reps
    return "OK"


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for synthesized results only.
    DeepSeek usage gives exact counts for prompts/completions."""
    return max(1, len(text) // 4)


def call_deepseek(messages: list, key: str) -> dict:
    body = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "max_tokens": 512,
    }
    r = requests.post(API_URL,
                      headers={"Authorization": f"Bearer {key}",
                               "Content-Type": "application/json"},
                      json=body, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(
            f"DeepSeek {r.status_code} at turn with {len(messages)} msgs: "
            f"{r.text[:500]}")
    return r.json()


def run_one_task(task: dict, key: str, rng: random.Random) -> dict:
    """Run one BFCL task as a ReAct loop on DeepSeek, recording the trace."""
    # task["question"] is [[{role,content}]] (BFCL wraps the user turn)
    q = task["question"][0][0]["content"]
    messages = [{"role": "system",
                 "content": "You are a research assistant. Use the web_search "
                            "and retrieve_page tools to answer multi-hop "
                            "questions. Be efficient: search at most 4-5 times, "
                            "then give a final concise answer. Do not keep "
                            "searching once you have enough information."},
                {"role": "user", "content": q}]
    turns = []
    for turn_idx in range(MAX_TURNS):
        resp = call_deepseek(messages, key)
        choice = resp["choices"][0]
        msg = choice["message"]
        usage = resp.get("usage", {})
        turn = {
            "turn_idx": turn_idx,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": usage.get("completion_tokens_details",
                                          {}).get("reasoning_tokens", 0),
            "assistant_text": msg.get("content") or "",
            "finish_reason": choice.get("finish_reason"),
        }
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            # No more tool calls -> agent is done (final answer in content)
            turns.append(turn)
            break
        # Record each tool call this turn (BFCL web search typically one/turn)
        for tc in tool_calls:
            fn = tc["function"]
            name = fn["name"]
            try:
                args = json.loads(fn["arguments"])
            except Exception:
                args = {"raw": fn["arguments"]}
            latency = sample_latency(rng)
            result = synthesize_result(name, args, rng)
            turn["tool_name"] = name
            turn["tool_args"] = args
            turn["tool_latency_s"] = latency
            turn["tool_result_tokens"] = estimate_tokens(result)
            # Append the assistant tool-call message and the tool result,
            # so the context accumulates (this is the prefix-reuse case).
            messages.append({
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": [{
                    "id": tc["id"], "type": "function",
                    "function": {"name": name, "arguments": fn["arguments"]},
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })
        turns.append(turn)
    return {
        "task_id": task["id"],
        "question": q,
        "turns": turns,
        "total_turns": len(turns),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="benchmarks/mimir/data/BFCL_v4_web_search.json")
    ap.add_argument("--out", default="benchmarks/mimir/traces/sample")
    ap.add_argument("--num-tasks", type=int, default=3)
    ap.add_argument("--start", type=int, default=0, help="task index to start at")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY env var not set")

    tasks = [json.loads(l) for l in open(args.data) if l.strip()]
    selected = tasks[args.start:args.start + args.num_tasks]
    rng = random.Random(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for i, task in enumerate(selected):
        t0 = time.time()
        try:
            trace = run_one_task(task, key, rng)
        except Exception as e:
            print(f"[{i}] task {task['id']} FAILED: {e}")
            continue
        trace["wall_time_s"] = round(time.time() - t0, 2)
        tf = out_dir / f"{task['id']}.json"
        tf.write_text(json.dumps(trace, ensure_ascii=False, indent=2))
        # aggregate stats
        prompt_tokens = [t["prompt_tokens"] for t in trace["turns"]]
        lats = [t.get("tool_latency_s", 0) for t in trace["turns"]]
        summary.append({
            "task_id": task["id"],
            "turns": trace["total_turns"],
            "final_prompt_tokens": prompt_tokens[-1] if prompt_tokens else 0,
            "max_prompt_tokens": max(prompt_tokens) if prompt_tokens else 0,
            "sum_tool_latency_s": round(sum(lats), 2),
        })
        print(f"[{i}] {task['id']}: turns={trace['total_turns']} "
              f"final_prompt={prompt_tokens[-1] if prompt_tokens else 0} "
              f"max_prompt={max(prompt_tokens) if prompt_tokens else 0} "
              f"sum_lat={round(sum(lats),2)}s wall={trace['wall_time_s']}s")

    (out_dir / "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
