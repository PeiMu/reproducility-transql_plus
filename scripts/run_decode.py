"""
Measure TranSQL+ decode latency (time per output token) with KV cache.

Paper Section 5: evaluates decode latency after prefill.

After prefill, K_rope and V tables persist as the KV cache. Each decode step
processes a single new token, appends its K/V to the cache, and runs attention
+ FFN for that token.

Measurement protocol (see reproduction_note.md):
  1. Open DuckDB connection (read_only=False for INSERT INTO KV cache)
  2. Run prefill (warms buffer pool)
  3. 2 warmup decode steps (discard)
  4. N measured decode steps
  5. Report: mean, std of measured steps (excluding warmup)

Usage:
    python scripts/run_decode.py \
        --db-path weights.duckdb \
        --prompts-dir prompts \
        --output results/decode.json \
        [--num-layers 32] \
        [--decode-steps 49] \
        [--warmup 2] \
        [--lengths 25 50 100 200]
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


WARMUP_DECODE_STEPS = 2
MEASURED_DECODE_STEPS = 49


def get_peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def get_db_size_gb(db_path: str) -> float:
    return os.path.getsize(db_path) / (1024 ** 3)


def measure_decode(
    db_path: str,
    config: ModelConfig,
    prompt_path: str,
    *,
    decode_steps: int = MEASURED_DECODE_STEPS,
    warmup: int = WARMUP_DECODE_STEPS,
    use_pivot: bool = True,
    pivot_width: int = 0,
    subquery_width: int = 0,
    memory_limit: str | None = None,
    threads: int | None = None,
    temp_directory: str | None = None,
) -> dict:
    """Measure decode latency for one prompt length."""

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

    # Decode needs write access (INSERT INTO KV cache)
    runner = TranSQLRunner(
        db_path, config, postopt=opts, read_only=False,
        memory_limit=memory_limit,
        threads=threads,
        temp_directory=temp_directory,
    )
    runner.init()

    # One-time weight pivot cost (Decision D9). Not part of measured latency.
    pivot_setup_time = runner.pivot_setup_time_s
    if pivot_setup_time > 0:
        print(f"  Pivot setup (once, D9): {pivot_setup_time:.3f}s")

    # Prefill (also warms buffer pool)
    print(f"  Prefill (seq_len={seq_len})...")
    t0 = time.perf_counter()
    prefill_result = runner.run_prefill(token_ids)
    prefill_time = time.perf_counter() - t0
    print(f"  Prefill: {prefill_time:.3f}s "
          f"({seq_len / prefill_time:.2f} tok/s)")

    # Get first decode token (greedy argmax from prefill logits)
    current_token = runner.get_logits_argmax()

    total_steps = warmup + decode_steps
    all_latencies = []

    for step in range(total_steps):
        pos = seq_len + step
        t0 = time.perf_counter()
        result = runner.run_decode_step(current_token, pos)
        dt = time.perf_counter() - t0
        all_latencies.append(dt)

        # Get next token for autoregressive generation
        current_token = runner.get_logits_argmax()

        phase = "warmup" if step < warmup else "measure"
        if step < warmup or (step - warmup) % 10 == 0 or step == total_steps - 1:
            print(f"    [{phase}] step {step+1}/{total_steps}: "
                  f"{dt:.3f}s ({1.0/dt:.2f} tok/s)")

    runner.close()

    # Split warmup and measured
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
        "pivot_setup_time_s": pivot_setup_time,
        "warmup_steps": warmup,
        "measured_steps": decode_steps,
        "peak_rss_mb": get_peak_rss_mb(),
        "num_layers": config.num_layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure TranSQL+ decode latency"
    )
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--prompts-dir", default="prompts")
    parser.add_argument("--output", default="results/decode.json")
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--decode-steps", type=int, default=MEASURED_DECODE_STEPS)
    parser.add_argument("--warmup", type=int, default=WARMUP_DECODE_STEPS)
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[25, 50, 100, 200])
    parser.add_argument("--no-pivot", action="store_true")
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
    print(f"Protocol: {args.warmup} warmup + {args.decode_steps} measured "
          f"decode steps")
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
        result = measure_decode(
            args.db_path, config, prompt_path,
            decode_steps=args.decode_steps, warmup=args.warmup,
            use_pivot=not args.no_pivot,
            pivot_width=args.pivot_width,
            subquery_width=args.subquery_width,
            memory_limit=args.memory_limit,
            threads=args.threads,
            temp_directory=args.temp_directory,
        )
        results.append(result)
        print(f"  => Decode: {result['decode_latency_mean_s']:.3f}s/tok "
              f"(+/- {result['decode_latency_std_s']:.3f}s)  "
              f"{result['decode_throughput_tok_per_s']:.2f} tok/s  "
              f"RSS: {result['peak_rss_mb']:.0f} MB\n")

    output = {
        "db_path": args.db_path,
        "db_size_gb": db_size,
        "num_layers": args.num_layers,
        "chunk_size": args.chunk_size,
        "use_pivot": not args.no_pivot,
        "pivot_width": args.pivot_width,
        "subquery_width": args.subquery_width,
        "warmup_steps": args.warmup,
        "measured_steps": args.decode_steps,
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
