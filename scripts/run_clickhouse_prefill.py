"""
Measure TranSQL+ prefill latency on ClickHouse.

Companion to ``scripts/run_prefill.py`` (DuckDB); same measurement
protocol and output schema so ``scripts/collect_results.py`` can load
both files with a single loader pattern.

Protocol (reproduction_note.md Measurement Protocol):
  1. Open ClickHouse client
  2. Pre-pivot weight tables (one-time setup, timed separately)
  3. 2 warmup runs (discard)
  4. 3 measured runs
  5. Report: mean, std of 3 measured runs

Usage:
    python scripts/run_clickhouse_prefill.py \\
        --ch-host localhost --ch-port 8123 --ch-database default \\
        --prompts-dir prompts \\
        --output results/clickhouse_prefill.json \\
        [--num-layers 32] [--lengths 25 50 100 200] \\
        [--max-memory-usage 17179869184] [--max-threads 4]
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time

import numpy as np

from transql_plus.clickhouse.runner_ch import ClickHouseRunner
from transql_plus.config import ModelConfig
from transql_plus.postopt import PostOptOptions


WARMUP_RUNS = 2
MEASURED_RUNS = 3


def parse_bytes(s: str) -> int:
    """Parse ``"16GB"``, ``"512MB"``, ``"17179869184"``, etc. → bytes.

    Uses 1024-based units (``GB`` == ``GiB``) to match DuckDB's
    ``memory_limit`` convention so the same literal works across both
    backends. ClickHouse's ``max_memory_usage`` setting itself takes
    raw bytes; the conversion happens here.
    """
    s = s.strip().upper()
    mults = [
        ("TIB", 1024 ** 4), ("GIB", 1024 ** 3),
        ("MIB", 1024 ** 2), ("KIB", 1024),
        ("TB",  1024 ** 4), ("GB",  1024 ** 3),
        ("MB",  1024 ** 2), ("KB",  1024),
        ("T",   1024 ** 4), ("G",   1024 ** 3),
        ("M",   1024 ** 2), ("K",   1024),
        ("B",   1),
    ]
    for suf, m in mults:
        if s.endswith(suf):
            return int(float(s[:-len(suf)]) * m)
    return int(s)


def get_peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def get_db_size_gb(client) -> float:
    """Total bytes across all non-system tables in the current database."""
    rows = client.query(
        "SELECT sum(total_bytes) FROM system.tables "
        "WHERE database = currentDatabase() AND is_temporary = 0"
    ).result_rows
    if not rows or rows[0][0] is None:
        return 0.0
    return float(rows[0][0]) / (1024 ** 3)


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
    database: str,
    max_memory_usage: int | None,
    max_threads: int | None,
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

    runner = ClickHouseRunner(
        config=config, host=host, port=port, user=user, password=password,
        database=database,
        postopt=opts,
        max_memory_usage=max_memory_usage,
        max_threads=max_threads,
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
    parser.add_argument("--ch-host", default="localhost")
    parser.add_argument("--ch-port", type=int, default=8123)
    parser.add_argument("--ch-user", default="default")
    parser.add_argument("--ch-password", default="")
    parser.add_argument("--ch-database", default="default")
    parser.add_argument("--prompts-dir", default="prompts")
    parser.add_argument("--output", default="results/clickhouse_prefill.json")
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[25, 50, 100, 200])
    parser.add_argument("--repeat", type=int, default=MEASURED_RUNS)
    parser.add_argument("--warmup", type=int, default=WARMUP_RUNS)
    parser.add_argument("--no-pivot", action="store_true")
    parser.add_argument("--pivot-width", type=int, default=0)
    parser.add_argument("--subquery-width", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=32)
    # Paper's hardware: AWS c7.2xlarge = 4 cores, 16 GB RAM.
    parser.add_argument("--max-memory-usage", type=parse_bytes, default=None,
                        help="ClickHouse max_memory_usage. Accepts raw "
                             "bytes or a suffix (e.g. '16GB'). Paper: 16GB.")
    parser.add_argument("--max-threads", type=int, default=None,
                        help="ClickHouse max_threads (paper uses 4)")
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

    import clickhouse_connect
    probe = clickhouse_connect.get_client(
        host=args.ch_host, port=args.ch_port,
        username=args.ch_user, password=args.ch_password,
        database=args.ch_database,
    )
    db_size = get_db_size_gb(probe)
    probe.close()

    print(f"ClickHouse: {args.ch_host}:{args.ch_port}/{args.ch_database} "
          f"({db_size:.2f} GB)")
    print(f"Config: {args.num_layers} layers, chunk_size={args.chunk_size}")
    print(f"Protocol: {args.warmup} warmup + {args.repeat} measured runs")
    if args.max_memory_usage or args.max_threads:
        print(f"ClickHouse: max_memory_usage={args.max_memory_usage} "
              f"max_threads={args.max_threads}")
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
            host=args.ch_host, port=args.ch_port,
            user=args.ch_user, password=args.ch_password,
            database=args.ch_database,
            max_memory_usage=args.max_memory_usage,
            max_threads=args.max_threads,
        )
        results.append(result)
        print(f"  => Mean: {result['prefill_latency_mean_s']:.3f}s "
              f"(+/- {result['prefill_latency_std_s']:.3f}s)  "
              f"{result['prefill_throughput_tok_per_s']:.2f} tok/s  "
              f"RSS: {result['peak_rss_mb']:.0f} MB\n")

    output = {
        "ch_host": args.ch_host,
        "ch_port": args.ch_port,
        "ch_database": args.ch_database,
        "db_size_gb": db_size,
        "num_layers": args.num_layers,
        "chunk_size": args.chunk_size,
        "use_pivot": not args.no_pivot,
        "pivot_width": args.pivot_width,
        "subquery_width": args.subquery_width,
        "warmup_runs": args.warmup,
        "measured_runs": args.repeat,
        "max_memory_usage": args.max_memory_usage,
        "max_threads": args.max_threads,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
