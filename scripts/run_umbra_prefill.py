"""
Measure TranSQL+ prefill latency on Umbra.

Companion to ``scripts/run_prefill.py`` (DuckDB); same measurement
protocol and output schema.

Protocol (reproduction_note.md Measurement Protocol):
  1. Open Umbra connection (psycopg2, port 15432)
  2. Pre-pivot weight tables (one-time setup, timed separately)
  3. 2 warmup runs (discard)
  4. 3 measured runs
  5. Report: mean, std of 3 measured runs

Usage:
    python scripts/run_umbra_prefill.py \\
        --host localhost --port 15432 \\
        --prompts-dir prompts \\
        --output results/umbra_prefill.json \\
        [--num-layers 32] [--lengths 25 50 100 200]
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time

import numpy as np

from transql_plus.config import ModelConfig
from transql_plus.postopt import PostOptOptions
from transql_plus.umbra.runner_umbra import UmbraRunner


WARMUP_RUNS = 2
MEASURED_RUNS = 3


def get_peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def measure_prefill(
    config: ModelConfig,
    prompt_path: str,
    *,
    repeat: int = MEASURED_RUNS,
    warmup: int = WARMUP_RUNS,
    use_pivot: bool = True,
    pivot_width: int = 0,
    subquery_width: int = 0,
    host: str,
    port: int,
    user: str,
    password: str,
) -> dict:
    with open(prompt_path) as f:
        prompt = json.load(f)
    token_ids = prompt["token_ids"]
    seq_len = len(token_ids)

    opts = PostOptOptions(
        row2col_pivot=use_pivot,
        cte_merge=True,
        table_fusion=True,
        pivot_width=pivot_width,
        subquery_width=subquery_width,
    ) if use_pivot else None

    runner = UmbraRunner(
        config=config, host=host, port=port,
        user=user, password=password,
        postopt=opts,
    )
    runner.init()

    if runner.pivot_setup_time_s > 0:
        print(f"  Pivot setup (once, D9): "
              f"{runner.pivot_setup_time_s:.3f}s")

    print(f"  Warmup: {warmup} runs...")
    for w in range(warmup):
        result = runner.run_prefill(token_ids)
        print(f"    warmup {w+1}: {result.latency_s:.3f}s (discarded)")

    latencies = []
    for r in range(repeat):
        result = runner.run_prefill(token_ids)
        latencies.append(result.latency_s)
        throughput = seq_len / result.latency_s
        print(f"    run {r+1}: {result.latency_s:.3f}s "
              f"({throughput:.2f} tok/s)")

    runner.close()

    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    mean_tput = seq_len / mean_lat

    return {
        "prompt_length": seq_len,
        "prefill_latencies_s": latencies,
        "prefill_latency_mean_s": mean_lat,
        "prefill_latency_std_s": std_lat,
        "prefill_throughput_tok_per_s": mean_tput,
        "pivot_setup_time_s": runner.pivot_setup_time_s,
        "peak_rss_mb": get_peak_rss_mb(),
        "num_layers": config.num_layers,
        "step_count": result.step_count,
        "warmup_runs": warmup,
        "measured_runs": repeat,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=15432)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="umbra")
    parser.add_argument("--prompts-dir", default="prompts")
    parser.add_argument("--output", default="results/umbra_prefill.json")
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[25, 50, 100, 200])
    parser.add_argument("--repeat", type=int, default=MEASURED_RUNS)
    parser.add_argument("--warmup", type=int, default=WARMUP_RUNS)
    parser.add_argument("--no-pivot", action="store_true")
    parser.add_argument("--pivot-width", type=int, default=0)
    parser.add_argument("--subquery-width", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()

    config = ModelConfig.llama3_8b(chunk_size=args.chunk_size)
    config = ModelConfig(
        hidden_dim=config.hidden_dim,
        num_q_heads=config.num_q_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        ffn_dim=config.ffn_dim,
        num_layers=args.num_layers,
        vocab_size=config.vocab_size,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        max_seq_len=config.max_seq_len,
        chunk_size=config.chunk_size,
    )

    print(f"Umbra: {args.host}:{args.port}")
    print(f"Config: {args.num_layers} layers, chunk_size={args.chunk_size}")
    print(f"Protocol: {args.warmup} warmup + {args.repeat} measured runs")
    print()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results = []

    for length in args.lengths:
        prompt_path = os.path.join(args.prompts_dir, f"prompt_{length}.json")
        if not os.path.exists(prompt_path):
            print(f"SKIP: {prompt_path} not found")
            continue

        print(f"Prompt length {length}:")
        result = measure_prefill(
            config, prompt_path,
            repeat=args.repeat, warmup=args.warmup,
            use_pivot=not args.no_pivot,
            pivot_width=args.pivot_width,
            subquery_width=args.subquery_width,
            host=args.host, port=args.port,
            user=args.user, password=args.password,
        )
        results.append(result)
        print(f"  => Mean: {result['prefill_latency_mean_s']:.3f}s "
              f"(+/- {result['prefill_latency_std_s']:.3f}s)  "
              f"{result['prefill_throughput_tok_per_s']:.2f} tok/s  "
              f"RSS: {result['peak_rss_mb']:.0f} MB\n")

    output = {
        "umbra_host": args.host,
        "umbra_port": args.port,
        "num_layers": args.num_layers,
        "chunk_size": args.chunk_size,
        "use_pivot": not args.no_pivot,
        "pivot_width": args.pivot_width,
        "subquery_width": args.subquery_width,
        "warmup_runs": args.warmup,
        "measured_runs": args.repeat,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
