"""
Measure TranSQL+ prefill latency (time to first token).

Paper Section 5: evaluates prefill at prompt lengths 25, 50, 100, 200 tokens.

Measurement protocol (see reproduction_note.md):
  1. Open DuckDB connection (once)
  2. Pre-pivot weight tables (one-time setup, timed separately)
  3. 2 warmup runs (discard) — warms DuckDB buffer pool
  4. 3 measured runs
  5. Report: mean, std of 3 measured runs

Usage:
    python scripts/run_prefill.py \
        --db-path weights.duckdb \
        --prompts-dir prompts \
        --output results/prefill.json \
        [--num-layers 32] \
        [--lengths 25 50 100 200] \
        [--repeat 3] [--warmup 2]
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
from transql_plus.runner import TranSQLRunner


WARMUP_RUNS = 2
MEASURED_RUNS = 3


def get_peak_rss_mb() -> float:
    """Peak RSS in MB (Linux: maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def get_db_size_gb(db_path: str) -> float:
    return os.path.getsize(db_path) / (1024 ** 3)


def measure_prefill(
    db_path: str,
    config: ModelConfig,
    prompt_path: str,
    *,
    repeat: int = MEASURED_RUNS,
    warmup: int = WARMUP_RUNS,
    use_pivot: bool = True,
    pivot_width: int = 0,
    subquery_width: int = 0,
    memory_limit: str | None = None,
    threads: int | None = None,
    temp_directory: str | None = None,
) -> dict:
    """Measure prefill latency for one prompt length."""

    with open(prompt_path) as f:
        prompt = json.load(f)
    token_ids = prompt["token_ids"]
    seq_len = len(token_ids)

    # Build runner with postopt (pivot + stable softmax for real models)
    opts = PostOptOptions(
        row2col_pivot=use_pivot,
        cte_merge=True,
        table_fusion=True,
        pivot_width=pivot_width,
        subquery_width=subquery_width,
    ) if use_pivot else None

    runner = TranSQLRunner(
        db_path, config, postopt=opts, read_only=True,
        memory_limit=memory_limit,
        threads=threads,
        temp_directory=temp_directory,
    )
    runner.init()

    # One-time weight pivot cost (Decision D9). Reported separately so it
    # is not conflated with measured per-run prefill latency.
    pivot_setup_time = runner.pivot_setup_time_s
    if pivot_setup_time > 0:
        print(f"  Pivot setup (once, D9): {pivot_setup_time:.3f}s")

    # Warmup runs (discard)
    print(f"  Warmup: {warmup} runs...")
    for w in range(warmup):
        result = runner.run_prefill(token_ids)
        print(f"    warmup {w+1}: {result.latency_s:.3f}s (discarded)")

    # Measured runs
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
        "pivot_setup_time_s": pivot_setup_time,
        "peak_rss_mb": get_peak_rss_mb(),
        "num_layers": config.num_layers,
        "step_count": result.step_count,
        "warmup_runs": warmup,
        "measured_runs": repeat,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure TranSQL+ prefill latency"
    )
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--prompts-dir", default="prompts")
    parser.add_argument("--output", default="results/prefill.json")
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[25, 50, 100, 200])
    parser.add_argument("--repeat", type=int, default=MEASURED_RUNS)
    parser.add_argument("--warmup", type=int, default=WARMUP_RUNS)
    parser.add_argument("--no-pivot", action="store_true",
                        help="Disable ROW2COL pivot (baseline comparison)")
    parser.add_argument("--pivot-width", type=int, default=0,
                        help="ROW2COL pivot_width (§4.3). 0 = single PIVOT "
                             "over all chunks (default; required for weight "
                             "pivot caching in runner.py). Tune via "
                             "scripts/tune_pivot.py.")
    parser.add_argument("--subquery-width", type=int, default=0,
                        help="ROW2COL subquery_width (§4.3). 0 = 1 dot-product "
                             "per CROSS JOIN CTE (paper example pattern). "
                             "Tune via scripts/tune_pivot.py.")
    parser.add_argument("--chunk-size", type=int, default=32)
    # Paper's hardware: AWS c7.2xlarge = 4 cores, 16 GB RAM.
    # Use --memory-limit 16GB --threads 4 to reproduce paper's constraint.
    parser.add_argument("--memory-limit", default=None,
                        help="DuckDB memory_limit (e.g. '16GB'). "
                             "Paper uses 16GB RAM — set matching value here.")
    parser.add_argument("--threads", type=int, default=None,
                        help="DuckDB threads (paper uses 4)")
    parser.add_argument("--temp-directory", default=None,
                        help="DuckDB temp_directory for spill files")
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

    db_size = get_db_size_gb(args.db_path)
    print(f"DuckDB: {args.db_path} ({db_size:.2f} GB)")
    print(f"Config: {args.num_layers} layers, chunk_size={args.chunk_size}")
    print(f"Protocol: {args.warmup} warmup + {args.repeat} measured runs")
    if args.memory_limit or args.threads or args.temp_directory:
        print(f"DuckDB: memory_limit={args.memory_limit} "
              f"threads={args.threads} temp_directory={args.temp_directory}")
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
            args.db_path, config, prompt_path,
            repeat=args.repeat, warmup=args.warmup,
            use_pivot=not args.no_pivot,
            pivot_width=args.pivot_width,
            subquery_width=args.subquery_width,
            memory_limit=args.memory_limit,
            threads=args.threads,
            temp_directory=args.temp_directory,
        )
        results.append(result)
        print(f"  => Mean: {result['prefill_latency_mean_s']:.3f}s "
              f"(+/- {result['prefill_latency_std_s']:.3f}s)  "
              f"{result['prefill_throughput_tok_per_s']:.2f} tok/s  "
              f"RSS: {result['peak_rss_mb']:.0f} MB\n")

    # Add metadata
    output = {
        "db_path": args.db_path,
        "db_size_gb": db_size,
        "num_layers": args.num_layers,
        "chunk_size": args.chunk_size,
        "use_pivot": not args.no_pivot,
        "pivot_width": args.pivot_width,
        "subquery_width": args.subquery_width,
        "warmup_runs": args.warmup,
        "measured_runs": args.repeat,
        "memory_limit": args.memory_limit,
        "threads": args.threads,
        "temp_directory": args.temp_directory,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
