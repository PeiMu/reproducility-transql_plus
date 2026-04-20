"""
Measure TranSQL+ decode latency on ClickHouse.

Companion to ``scripts/run_decode.py`` (DuckDB). Same protocol, same
output schema — ``scripts/collect_results.py`` consumes both.

Protocol (reproduction_note.md Measurement Protocol):
  1. Connect to ClickHouse
  2. Run prefill (warms buffer pool)
  3. 2 warmup decode steps (discard)
  4. 49 measured decode steps
  5. Report: mean, std of measured steps
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


WARMUP_DECODE_STEPS = 2
MEASURED_DECODE_STEPS = 49


def parse_bytes(s: str) -> int:
    """``"16GB"`` → bytes. 1024-based (GB == GiB, matches DuckDB)."""
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


def measure_decode(
    config: ModelConfig,
    prompt_path: str,
    *,
    decode_steps: int = MEASURED_DECODE_STEPS,
    warmup: int = WARMUP_DECODE_STEPS,
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
        database=database, postopt=opts,
        max_memory_usage=max_memory_usage,
        max_threads=max_threads,
    )
    runner.init()

    if runner.pivot_setup_time_s > 0:
        print(f"  Pivot setup (once, D9): "
              f"{runner.pivot_setup_time_s:.3f}s")

    print(f"  Prefill (seq_len={seq_len})...")
    t0 = time.perf_counter()
    runner.run_prefill(token_ids)
    prefill_time = time.perf_counter() - t0
    print(f"  Prefill: {prefill_time:.3f}s "
          f"({seq_len / prefill_time:.2f} tok/s)")

    current_token = runner.get_logits_argmax()

    total_steps = warmup + decode_steps
    all_latencies = []

    for step in range(total_steps):
        pos = seq_len + step
        t0 = time.perf_counter()
        runner.run_decode_step(current_token, pos)
        dt = time.perf_counter() - t0
        all_latencies.append(dt)

        current_token = runner.get_logits_argmax()

        phase = "warmup" if step < warmup else "measure"
        if step < warmup or (step - warmup) % 10 == 0 or step == total_steps - 1:
            print(f"    [{phase}] step {step+1}/{total_steps}: "
                  f"{dt:.3f}s ({1.0/dt:.2f} tok/s)")

    runner.close()

    measured = all_latencies[warmup:]
    mean_lat = float(np.mean(measured))
    std_lat = float(np.std(measured))

    return {
        "prompt_length": seq_len,
        "prefill_latency_s": prefill_time,
        "prefill_throughput_tok_per_s": seq_len / prefill_time,
        "decode_latencies_s": measured,
        "decode_latency_mean_s": mean_lat,
        "decode_latency_std_s": std_lat,
        "decode_throughput_tok_per_s": 1.0 / mean_lat,
        "pivot_setup_time_s": runner.pivot_setup_time_s,
        "warmup_steps": warmup,
        "measured_steps": decode_steps,
        "peak_rss_mb": get_peak_rss_mb(),
        "num_layers": config.num_layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ch-host", default="localhost")
    parser.add_argument("--ch-port", type=int, default=8123)
    parser.add_argument("--ch-user", default="default")
    parser.add_argument("--ch-password", default="")
    parser.add_argument("--ch-database", default="default")
    parser.add_argument("--prompts-dir", default="prompts")
    parser.add_argument("--output", default="results/clickhouse_decode.json")
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--decode-steps", type=int, default=MEASURED_DECODE_STEPS)
    parser.add_argument("--warmup", type=int, default=WARMUP_DECODE_STEPS)
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[25, 50, 100, 200])
    parser.add_argument("--no-pivot", action="store_true")
    parser.add_argument("--pivot-width", type=int, default=0)
    parser.add_argument("--subquery-width", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--max-memory-usage", type=parse_bytes, default=None,
                        help="ClickHouse max_memory_usage. Accepts raw "
                             "bytes or a suffix (e.g. '16GB').")
    parser.add_argument("--max-threads", type=int, default=None)
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

    print(f"ClickHouse: {args.ch_host}:{args.ch_port}/{args.ch_database}")
    print(f"Config: {args.num_layers} layers, chunk_size={args.chunk_size}")
    print(f"Protocol: {args.warmup} warmup + {args.decode_steps} measured "
          f"decode steps")
    print()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results = []

    for length in args.lengths:
        prompt_path = os.path.join(args.prompts_dir, f"prompt_{length}.json")
        if not os.path.exists(prompt_path):
            print(f"SKIP: {prompt_path} not found")
            continue

        print(f"Prompt length {length}:")
        result = measure_decode(
            config, prompt_path,
            decode_steps=args.decode_steps, warmup=args.warmup,
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
        print(f"  => Decode: {result['decode_latency_mean_s']:.3f}s/tok "
              f"(+/- {result['decode_latency_std_s']:.3f}s)  "
              f"{result['decode_throughput_tok_per_s']:.2f} tok/s  "
              f"RSS: {result['peak_rss_mb']:.0f} MB\n")

    output = {
        "ch_host": args.ch_host,
        "ch_port": args.ch_port,
        "ch_database": args.ch_database,
        "num_layers": args.num_layers,
        "chunk_size": args.chunk_size,
        "use_pivot": not args.no_pivot,
        "pivot_width": args.pivot_width,
        "subquery_width": args.subquery_width,
        "warmup_steps": args.warmup,
        "measured_steps": args.decode_steps,
        "max_memory_usage": args.max_memory_usage,
        "max_threads": args.max_threads,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
