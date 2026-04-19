"""
Tuning script for ROW2COL pivot configuration.

Paper §4.3: "This creates a trade-off: too many projections in one query
can exceed parallelism, while too many subqueries increase I/O. To study
this balance, we benchmark speedups over unoptimized SQL across
#projection × #subquery configurations and CPU core counts."

Tested matrices (adapted from paper §4.3 — Llama3-8B shapes):
  For every prompt length L in --lengths, all four projection shapes:
    - Q/O:     act(L, 4096)  × wt(4096,  4096)   contracted=hidden
    - K/V:     act(L, 4096)  × wt(1024,  4096)   contracted=hidden
    - gate/up: act(L, 4096)  × wt(14336, 4096)   contracted=hidden
    - down:    act(L, 14336) × wt(4096,  14336)  contracted=ffn

Realistic `out_dim` matters: CROSS JOIN cardinality = act_rows × weight_rows,
so tuning with a tiny synthetic weight (e.g. the earlier 25×128 cap) makes
pivot setup overhead dominate the measurement and hides the §4.3 speedup.

Alignment with run_prefill.py (so the tuned pair transfers correctly):
  - Weight pivots cached outside the timed loop (D9), like runner.py.
  - Non-cached pivoted steps merged via _emit_cte_block → one WITH block
    per run (§4.1 cte_merge, what run_prefill actually executes).
  - should_pivot (postopt.py:380) skipped matrices are reported as SKIP —
    run_prefill runs those MatMuls via the baseline path, not pivoted.
  - `--memory-limit` / `--temp-directory` / `--threads` match run_prefill
    so tuning happens under the same DuckDB constraints.

Usage:
    python scripts/tune_pivot.py \
        [--chunk-size 32] [--threads 4] \
        [--lengths 25,50,100,200] [--repeats 3] \
        [--memory-limit 16GB] [--temp-directory ./duckdb_tmp]
"""

from __future__ import annotations

import argparse
import itertools
import time

import duckdb
import numpy as np


def load_2d(con: duckdb.DuckDBPyConnection,
            table_name: str, arr: np.ndarray,
            chunk_size: int) -> None:
    """Load a 2D numpy array into a chunked DuckDB table."""
    out_dim, in_dim = arr.shape
    assert in_dim % chunk_size == 0
    n_chunks = in_dim // chunk_size

    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(
        f"CREATE TABLE {table_name} "
        f"(row_index INTEGER, chunk_index INTEGER, v FLOAT[{chunk_size}], "
        f"PRIMARY KEY (row_index, chunk_index))"
    )

    rows = []
    for r in range(out_dim):
        for c in range(n_chunks):
            offset = c * chunk_size
            chunk = arr[r, c * chunk_size:(c + 1) * chunk_size].tolist()
            rows.append((r, offset, chunk))

    con.executemany(
        f"INSERT INTO {table_name} VALUES (?, ?, ?::FLOAT[{chunk_size}])",
        rows,
    )


def run_baseline(con, act_table, weight_table, out_table, chunk_size):
    """Standard MatMul: JOIN + SUM(list_dot_product) + re-chunk."""
    cs = str(chunk_size)
    dp = out_table + "_dp"

    con.execute(f"DROP TABLE IF EXISTS {dp}")
    con.execute(f"DROP TABLE IF EXISTS {out_table}")

    con.execute(
        f"CREATE TEMP TABLE {dp} AS ("
        f"SELECT a.row_index AS act_row, w.row_index AS out_col, "
        f"SUM(list_dot_product(a.v, w.v)) AS val "
        f"FROM {act_table} a JOIN {weight_table} w "
        f"ON a.chunk_index = w.chunk_index "
        f"GROUP BY a.row_index, w.row_index)"
    )

    con.execute(
        f"CREATE TEMP TABLE {out_table} AS ("
        f"SELECT act_row AS row_index, "
        f"out_col - (out_col % {cs}) AS chunk_index, "
        f"array_agg(val ORDER BY out_col) AS v "
        f"FROM {dp} "
        f"GROUP BY act_row, out_col - (out_col % {cs}))"
    )


def materialize_weight_pivots(con, weight_table, n_chunks, chunk_size,
                              pivot_width):
    """Pre-pivot the weight table for a given pivot_width (D9 cache).

    Mirrors `runner.py`'s one-time weight pivot. Returns the set of table
    names that were created so `run_pivoted` can skip recreating them.
    Timed outside the measurement loop — matches `pivot_setup_time_s`
    accounting in run_prefill/run_decode/run_perplexity.
    """
    from transql_plus.postopt import pivoted_matmul_sql

    steps = pivoted_matmul_sql(
        "__dummy_act__", weight_table, "__dummy_out__",
        n_chunks, chunk_size, pivot_width, subquery_width=1,
    )

    # Weight-pivot step names contain f"{weight_table}_piv" (optionally _gN)
    cached: set[str] = set()
    for sql, name in steps:
        if name.startswith(f"{weight_table}_piv"):
            con.execute(f"DROP TABLE IF EXISTS {name}")
            con.execute(f"CREATE TEMP TABLE {name} AS ({sql})")
            cached.add(name)
    return cached


def run_pivoted(con, act_table, weight_table, out_table,
                n_chunks, chunk_size, pivot_width, subquery_width,
                cached_weight_pivots=None):
    """Pivoted MatMul: PIVOT → CROSS JOIN → POSITIONAL JOIN → re-chunk.

    Applies CTE merge (§4.1) to the non-cached steps, matching what
    `postopt_dag_to_sql` does when `cte_merge=True` in run_prefill.
    """
    from transql_plus.postopt import _emit_cte_block, pivoted_matmul_sql

    steps = pivoted_matmul_sql(
        act_table, weight_table, out_table,
        n_chunks, chunk_size, pivot_width, subquery_width,
    )

    cache = cached_weight_pivots or set()

    # Drop any existing temp tables from a previous run (skip cached weights)
    for _, name in steps:
        if name in cache:
            continue
        con.execute(f"DROP TABLE IF EXISTS {name}")

    # Filter cached weight pivots out, then CTE-merge the rest into a
    # single WITH block (§4.1). This mirrors postopt_dag_to_sql's output
    # for a single at-output MatMul and removes the per-step CREATE TEMP
    # TABLE overhead that would otherwise only appear in this script.
    non_cached = [(sql, name) for sql, name in steps if name not in cache]
    if not non_cached:
        return
    merged_sql, merged_name = _emit_cte_block(non_cached)
    con.execute(f"CREATE TEMP TABLE {merged_name} AS ({merged_sql})")


def benchmark_one(con, act_table, weight_table, n_chunks, chunk_size,
                  pivot_width, subquery_width, warmup=1, repeats=3,
                  cached_weight_pivots=None):
    """Time a single pivoted MatMul configuration."""
    # Warmup
    for _ in range(warmup):
        run_pivoted(con, act_table, weight_table, "bench_out",
                    n_chunks, chunk_size, pivot_width, subquery_width,
                    cached_weight_pivots=cached_weight_pivots)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run_pivoted(con, act_table, weight_table, "bench_out",
                    n_chunks, chunk_size, pivot_width, subquery_width,
                    cached_weight_pivots=cached_weight_pivots)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return min(times), sum(times) / len(times)


def benchmark_baseline(con, act_table, weight_table, chunk_size,
                       warmup=1, repeats=3):
    """Time the baseline (unoptimized) MatMul."""
    for _ in range(warmup):
        run_baseline(con, act_table, weight_table, "bench_base", chunk_size)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run_baseline(con, act_table, weight_table, "bench_base", chunk_size)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return min(times), sum(times) / len(times)


def main():
    parser = argparse.ArgumentParser(
        description="Tune ROW2COL pivot #projection × #subquery (§4.3)")
    parser.add_argument("--chunk-size", type=int, default=32,
                        help="Chunk size — must match weights.duckdb's "
                             "stored chunk_size (default: 32)")
    parser.add_argument("--threads", type=str, default="4",
                        help="Comma-separated thread counts to test "
                             "(default: 4 — matches paper c7.2xlarge)")
    parser.add_argument("--lengths", type=str, default="25,50,100,200",
                        help="Comma-separated prompt lengths to tune for "
                             "(default: 25,50,100,200 — matches "
                             "run_prefill.py default --lengths)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Timing repeats per config")
    parser.add_argument("--quick", action="store_true",
                        help="Test fewer configurations for faster results")
    # Paper's hardware: AWS c7.2xlarge = 4 cores, 16 GB RAM.
    # Align tuning with the full benchmark's DuckDB settings so the tuned
    # (pivot_width, subquery_width) transfers correctly to run_prefill.
    parser.add_argument("--memory-limit", default=None,
                        help="DuckDB memory_limit (e.g. '16GB'). Match "
                             "run_prefill.py for paper-faithful tuning.")
    parser.add_argument("--temp-directory", default=None,
                        help="DuckDB temp_directory for spill files. Match "
                             "run_prefill.py for paper-faithful tuning.")
    args = parser.parse_args()

    cs = args.chunk_size
    thread_list = [int(t) for t in args.threads.split(",")]
    lengths = [int(L) for L in args.lengths.split(",")]
    repeats = args.repeats

    # Llama3-8B projection shapes (paper §4.3, adapted).
    #   MatMul:  act(L, contracted)  ×  wt(out_dim, contracted)
    # Realistic out_dim is critical: CROSS JOIN cardinality scales with
    # act_rows × weight_rows, so a tiny weight out_dim hides the §4.3 speedup.
    matmul_shapes = [
        ("Q/O",     4096,  4096),   # contracted=hidden, out=hidden
        ("K/V",     4096,  1024),   # contracted=hidden, out=kv_dim
        ("gate/up", 4096, 14336),   # contracted=hidden, out=ffn
        ("down",   14336,  4096),   # contracted=ffn,    out=hidden
    ]
    test_cases = []
    for L in lengths:
        for label, contracted, out_dim in matmul_shapes:
            name = f"L={L} {label} (act {L}×{contracted}, wt {out_dim}×{contracted})"
            test_cases.append((name, L, contracted, out_dim))

    rng = np.random.default_rng(42)

    # Collect speedups keyed by (pw, sw) across all matrices + threads so
    # we can compute a global "best on average" recommendation at the end.
    all_speedups: dict[tuple[int, int], list[float]] = {}

    # Import here so `--help` doesn't pay the import cost
    from transql_plus.postopt import PostOptOptions, should_pivot
    default_opts = PostOptOptions()

    for name, m, contracted, out_dim in test_cases:
        # Ensure contracted dim is chunk-aligned
        assert contracted % cs == 0, (
            f"{name}: contracted={contracted} not divisible by chunk_size={cs}")
        n_chunks = contracted // cs

        print(f"\n{'='*70}")
        print(f"Matrix: {name}")
        print(f"  chunk_size={cs}, n_chunks={n_chunks}")
        print(f"{'='*70}")

        # Real runner's should_pivot heuristic (postopt.py:380) rejects
        # matrices with n_chunks > max_pivot_chunks (=128 by default).
        # Those MatMuls are NEVER pivoted in run_prefill — benchmarking
        # pivoted configs for them produces numbers the real runner
        # won't use. Skip with a clear notice.
        if not should_pivot(n_chunks, cs, default_opts):
            print(f"  SKIP: should_pivot heuristic rejects this matrix "
                  f"(n_chunks={n_chunks} > max_pivot_chunks="
                  f"{default_opts.max_pivot_chunks} "
                  f"or chunk_size={cs} > max_chunk_size="
                  f"{default_opts.max_chunk_size}). "
                  f"run_prefill runs this MatMul via the baseline path.")
            continue

        # Generate synthetic data at realistic Llama3-8B shapes
        act_np = rng.standard_normal((m, contracted)).astype(np.float32) * 0.01
        wt_np = rng.standard_normal((out_dim, contracted)).astype(np.float32) * 0.01

        # Pivot width configs: factors of n_chunks + full width
        if args.quick:
            pivot_widths = [n_chunks]
        else:
            pivot_widths = sorted({
                pw for pw in [1, 2, 4, 8, 16, 32, 64, n_chunks]
                if pw <= n_chunks
            })

        # Subquery widths: factors that make sense
        if args.quick:
            subquery_widths = [1, n_chunks]
        else:
            subquery_widths = sorted({
                sw for sw in [1, 2, 4, 8, 16, 32, 64]
                if sw <= n_chunks
            })

        for threads in thread_list:
            # Match runner.py's DuckDB init so tuning reflects the real
            # benchmark's constraints (memory_limit, temp_directory).
            duckdb_config: dict[str, str | int] = {"threads": threads}
            if args.memory_limit is not None:
                duckdb_config["memory_limit"] = args.memory_limit
            if args.temp_directory is not None:
                duckdb_config["temp_directory"] = args.temp_directory
            con = duckdb.connect(":memory:", config=duckdb_config)

            load_2d(con, "act", act_np, cs)
            load_2d(con, "wt", wt_np, cs)

            # Baseline
            base_min, base_avg = benchmark_baseline(
                con, "act", "wt", cs, repeats=repeats)
            print(f"\n  Threads={threads}, Baseline: "
                  f"min={base_min:.3f}s, avg={base_avg:.3f}s")
            print(f"  {'pivot_width':>12} {'subquery_width':>15} "
                  f"{'min_time':>10} {'avg_time':>10} {'speedup':>8}")
            print(f"  {'-'*60}")

            best_speedup = 0.0
            best_config = (0, 0)

            # Outer loop over pivot_width: weight pivot depends only on pw,
            # so materialize once per pw (D9 cache) and reuse across sw.
            for pw in pivot_widths:
                try:
                    cached = materialize_weight_pivots(
                        con, "wt", n_chunks, cs, pw)
                except Exception as e:
                    print(f"  {pw:>12} {'-':>15}  CACHE ERROR: {e}")
                    continue

                for sw in subquery_widths:
                    if sw > pw:
                        continue  # subquery_width > pivot_width makes no sense

                    try:
                        piv_min, piv_avg = benchmark_one(
                            con, "act", "wt", n_chunks, cs, pw, sw,
                            repeats=repeats,
                            cached_weight_pivots=cached)
                        speedup = base_avg / piv_avg if piv_avg > 0 else 0
                        print(f"  {pw:>12} {sw:>15} "
                              f"{piv_min:>10.3f} {piv_avg:>10.3f} "
                              f"{speedup:>7.2f}x")

                        if speedup > best_speedup:
                            best_speedup = speedup
                            best_config = (pw, sw)
                        all_speedups.setdefault((pw, sw), []).append(speedup)
                    except Exception as e:
                        print(f"  {pw:>12} {sw:>15}  ERROR: {e}")

                # Drop cached pivots before next pw to free memory
                for piv_name in cached:
                    con.execute(f"DROP TABLE IF EXISTS {piv_name}")

            print(f"\n  Best: pivot_width={best_config[0]}, "
                  f"subquery_width={best_config[1]}, "
                  f"speedup={best_speedup:.2f}x")

            con.close()

    # ── Global recommendation across all matrices + threads ──
    # A single (pw, sw) is passed to run_prefill.py for the full benchmark,
    # so we want the config with the highest geometric-mean speedup across
    # every tested (matrix, threads) combination.
    if all_speedups:
        import math
        print(f"\n{'='*70}")
        print(f"Global recommendation (geometric mean speedup across all "
              f"matrices + threads)")
        print(f"{'='*70}")
        scored = []
        for (pw, sw), speedups in all_speedups.items():
            if not speedups:
                continue
            geomean = math.exp(
                sum(math.log(s) for s in speedups if s > 0)
                / len(speedups)
            )
            scored.append(((pw, sw), geomean, min(speedups), max(speedups)))
        scored.sort(key=lambda x: x[1], reverse=True)

        print(f"  {'pivot_width':>12} {'subquery_width':>15} "
              f"{'geomean':>10} {'min':>8} {'max':>8}")
        print(f"  {'-'*60}")
        for (pw, sw), gm, mn, mx in scored[:10]:
            print(f"  {pw:>12} {sw:>15} {gm:>9.2f}x "
                  f"{mn:>7.2f}x {mx:>7.2f}x")
        best = scored[0]
        print(f"\nSuggested flags for run_prefill/run_decode/run_perplexity:")
        print(f"  --pivot-width {best[0][0]} --subquery-width {best[0][1]}")


if __name__ == "__main__":
    main()
