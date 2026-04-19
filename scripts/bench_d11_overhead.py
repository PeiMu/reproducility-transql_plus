"""Benchmark D11 overhead: fusion+materialize vs no-fusion.

Question answered by this script
--------------------------------
Is the post-prefill materialization of l{l}_v (D11 fix) negligible,
and is the chosen Option 1 actually better than Option 2 (disable
QKV fusion entirely)?

Variants
--------
A  fusion ON  + D11 materialize l{l}_v after prefill  (current default)
B  fusion OFF                                         (Option 2, rejected)

Both variants share the real-decode config by default:
  --pivot-width 32 --subquery-width 4
  --memory-limit 16GB --threads 4
  --temp-directory ./duckdb_tmp
  --num-layers 32

Per variant per prompt length the script measures
  pivot_setup_s    D9 one-time weight pivot cost (outside per-run latency)
  prefill_s        full DAG execution from input_tokens to logits
  d11_materialize_s   A only: CREATE TEMP TABLE l{l}_v ... × num_layers
  decode_mean_s    mean of `measure` decode steps after `warmup`
  session_total_s  prefill + d11 + (warmup + measure) × decode_mean

Then reports
  D11 overhead vs prefill        d11 / prefill (should be <<1%)
  Fusion benefit at prefill      B.prefill - A.prefill (expected positive)
  Session time delta             B.session - A.session (positive = D11 wins)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean

import numpy as np

from transql_plus.config import ModelConfig
from transql_plus.postopt import PostOptOptions
from transql_plus.runner import TranSQLRunner


def bench_one(
    *,
    db_path: str,
    cfg: ModelConfig,
    token_ids: list[int],
    fusion_on: bool,
    pivot_width: int,
    subquery_width: int,
    memory_limit: str | None,
    threads: int | None,
    temp_directory: str | None,
    warmup: int,
    measure: int,
) -> dict:
    """Run one (variant, prompt_length) benchmark and return timings."""
    opts = PostOptOptions(
        cte_merge=True,
        table_fusion=fusion_on,
        row2col_pivot=True,
        pivot_width=pivot_width,
        subquery_width=subquery_width,
    )
    runner = TranSQLRunner(
        db_path, cfg, postopt=opts, read_only=False,
        memory_limit=memory_limit,
        threads=threads,
        temp_directory=temp_directory,
    )
    runner.init()
    pivot_s = runner.pivot_setup_time_s

    # Prefill
    t0 = time.perf_counter()
    runner.run_prefill(token_ids)
    prefill_s = time.perf_counter() - t0

    # D11 materialize — direct timing.  Only meaningful when fusion is on;
    # with fusion off the method is a no-op (l{l}_v already exists).
    t0 = time.perf_counter()
    runner._materialize_fused_v_for_decode()
    d11_s = time.perf_counter() - t0
    runner._kv_cache_prepared = True  # prevent repeat inside decode step

    # Decode steps
    current = runner.get_logits_argmax()
    decode_times: list[float] = []
    for i in range(warmup + measure):
        pos = len(token_ids) + i
        t0 = time.perf_counter()
        runner.run_decode_step(current, pos)
        decode_times.append(time.perf_counter() - t0)
        current = runner.get_logits_argmax()

    runner.close()

    decode_measured = decode_times[warmup:]
    decode_mean = float(mean(decode_measured)) if decode_measured else 0.0
    decode_std = float(np.std(decode_measured)) if decode_measured else 0.0
    session_total = prefill_s + d11_s + sum(decode_times)

    return dict(
        pivot_s=pivot_s,
        prefill_s=prefill_s,
        d11_s=d11_s,
        decode_mean_s=decode_mean,
        decode_std_s=decode_std,
        session_total_s=session_total,
        num_decode_steps=len(decode_times),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db-path", required=True)
    p.add_argument("--prompts-dir", default="prompts")
    p.add_argument("--num-layers", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--pivot-width", type=int, default=32)
    p.add_argument("--subquery-width", type=int, default=4)
    p.add_argument("--lengths", type=int, nargs="+",
                   default=[25, 50, 100, 200])
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--measure", type=int, default=3)
    p.add_argument("--memory-limit", default="16GB")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--temp-directory", default="./duckdb_tmp")
    p.add_argument("--output", default="results/bench_d11_overhead.json")
    args = p.parse_args()

    cfg = ModelConfig.llama3_8b(chunk_size=args.chunk_size)
    cfg.num_layers = args.num_layers  # override

    print(f"DuckDB: {args.db_path}")
    print(f"Config: {args.num_layers} layers, chunk_size={args.chunk_size}")
    print(f"Pivot: width={args.pivot_width}, subquery={args.subquery_width}")
    print(f"Runtime: memory_limit={args.memory_limit} threads={args.threads} "
          f"temp_directory={args.temp_directory}")
    print(f"Protocol: warmup={args.warmup} + measure={args.measure} "
          f"decode steps per variant")
    print()

    prompts_dir = Path(args.prompts_dir)
    all_results: dict[str, dict] = {}

    for L in args.lengths:
        prompt_path = prompts_dir / f"prompt_{L}.json"
        if not prompt_path.exists():
            print(f"[skip] {prompt_path} not found")
            continue
        token_ids = json.loads(prompt_path.read_text())["token_ids"]

        print(f"=== Prompt length {L} ===")
        print(f"  [A] fusion ON + D11 materialize l{{l}}_v ...")
        a = bench_one(
            db_path=args.db_path, cfg=cfg, token_ids=token_ids,
            fusion_on=True,
            pivot_width=args.pivot_width,
            subquery_width=args.subquery_width,
            memory_limit=args.memory_limit, threads=args.threads,
            temp_directory=args.temp_directory,
            warmup=args.warmup, measure=args.measure,
        )
        print(f"      pivot_setup={a['pivot_s']:.2f}s  "
              f"prefill={a['prefill_s']:.2f}s  "
              f"d11={a['d11_s']*1000:.2f}ms  "
              f"decode_mean={a['decode_mean_s']:.3f}s "
              f"(+/- {a['decode_std_s']:.3f})")

        print(f"  [B] fusion OFF ...")
        b = bench_one(
            db_path=args.db_path, cfg=cfg, token_ids=token_ids,
            fusion_on=False,
            pivot_width=args.pivot_width,
            subquery_width=args.subquery_width,
            memory_limit=args.memory_limit, threads=args.threads,
            temp_directory=args.temp_directory,
            warmup=args.warmup, measure=args.measure,
        )
        print(f"      pivot_setup={b['pivot_s']:.2f}s  "
              f"prefill={b['prefill_s']:.2f}s  "
              f"decode_mean={b['decode_mean_s']:.3f}s "
              f"(+/- {b['decode_std_s']:.3f})")

        # Deltas
        d11_over_prefill = a["d11_s"] / a["prefill_s"] * 100.0
        fusion_benefit_s = b["prefill_s"] - a["prefill_s"]
        session_delta_s = b["session_total_s"] - a["session_total_s"]
        print(f"  --> D11 overhead vs A prefill : "
              f"{a['d11_s']*1000:.2f}ms  "
              f"({d11_over_prefill:.3f}% of prefill)")
        print(f"  --> Fusion benefit at prefill : "
              f"{fusion_benefit_s:+.3f}s (B-A, >0 means fusion wins)")
        print(f"  --> Session total delta       : "
              f"{session_delta_s:+.3f}s (B-A, >0 means D11 wins)")
        print()

        all_results[str(L)] = dict(A_fusion_on=a, B_fusion_off=b)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": vars(args),
        "results": all_results,
    }, indent=2))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
