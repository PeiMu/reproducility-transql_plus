"""
Tuning script for ROW2COL pivot_width on ClickHouse.

Companion to ``scripts/tune_pivot.py`` (DuckDB). Only ``pivot_width`` is
tuned here: the ClickHouse rewrite of paper §4.3 collapses the per-chunk
dot-product into a single ``SELECT`` (there is no POSITIONAL JOIN to
trade off against), so ``subquery_width`` is a no-op -- see
``transql_plus/clickhouse/postopt_ch.py::pivoted_matmul_sql``.

Paper §4.3: "This creates a trade-off: too many projections in one query
can exceed parallelism, while too many subqueries increase I/O." On
ClickHouse the subquery axis collapses, so the whole trade-off reduces
to the #projection (pivot_width) axis.

Matrix coverage mirrors tune_pivot.py (Llama3-8B shapes):
  Q/O:     act(L, 4096)  x wt(4096,  4096)
  K/V:     act(L, 4096)  x wt(1024,  4096)
  gate/up: act(L, 4096)  x wt(14336, 4096)
  down:    act(L, 14336) x wt(4096,  14336)

Alignment with run_clickhouse_prefill.py (so the tuned pivot_width
transfers correctly to the full benchmark):
  - Weight pivots materialised outside the timed loop (D9), matching
    ``ClickHouseRunner._pivot_weight_tables``.
  - Non-cached pivoted steps CTE-merged into one WITH block (§4.1),
    matching ``postopt_dag_to_sql_ch`` with ``cte_merge=True``.
  - ``should_pivot`` heuristic (postopt.py:380) applied -- rejected
    matrices reported as SKIP (run_clickhouse_prefill runs those via
    the baseline path).
  - ``--max-memory-usage`` / ``--max-threads`` forwarded as ClickHouse
    session settings so tuning happens under the paper's 16 GB /
    4-thread constraint.

Usage:
    python scripts/tune_pivot_clickhouse.py \\
        --ch-host localhost --ch-port 8123 --ch-database default \\
        [--chunk-size 32] [--max-threads 4] \\
        [--lengths 25,50,100,200] [--repeats 3] \\
        [--max-memory-usage 16GB]
"""

from __future__ import annotations

import argparse
import math
import time
import uuid

import clickhouse_connect
import numpy as np

from transql_plus.clickhouse.postopt_ch import (
    _emit_cte_block,
    pivot_sql,
    pivoted_matmul_sql,
)
from transql_plus.clickhouse.sql_templates_ch import _order_then_agg
from transql_plus.postopt import PostOptOptions, should_pivot


# ──────────────────────────────────────────────────────────────────────
# CLI helpers
# ──────────────────────────────────────────────────────────────────────

def parse_bytes(s: str) -> int:
    """``"16GB"`` -> bytes. 1024-based (matches run_clickhouse_prefill)."""
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


# ──────────────────────────────────────────────────────────────────────
# Table setup
# ──────────────────────────────────────────────────────────────────────

def load_2d(client, table_name: str, arr: np.ndarray,
            chunk_size: int) -> None:
    """Load a 2D numpy array into a chunked ClickHouse temp table.

    Temp tables (``ENGINE=Memory``) live for the lifetime of the client
    session, which matches how the real ClickHouse runner stages
    activations and mirrors the DuckDB tune_pivot loader's use of
    in-memory tables.
    """
    out_dim, in_dim = arr.shape
    assert in_dim % chunk_size == 0, (
        f"in_dim={in_dim} not divisible by chunk_size={chunk_size}")
    n_chunks = in_dim // chunk_size

    client.command(f"DROP TEMPORARY TABLE IF EXISTS {table_name}")
    client.command(
        f"CREATE TEMPORARY TABLE {table_name} ("
        f"row_index Int32, chunk_index Int32, v Array(Float32)"
        f") ENGINE=Memory"
    )

    rows = []
    for r in range(out_dim):
        for c in range(n_chunks):
            offset = c * chunk_size
            chunk = arr[r, c * chunk_size:(c + 1) * chunk_size].tolist()
            rows.append((r, offset, chunk))

    client.insert(table_name, rows,
                  column_names=["row_index", "chunk_index", "v"])


# ──────────────────────────────────────────────────────────────────────
# Baseline MatMul (no pivoting; mirrors matmul_sql in sql_templates_ch)
# ──────────────────────────────────────────────────────────────────────

def run_baseline(client, act_table: str, weight_table: str,
                 out_table: str, chunk_size: int) -> None:
    """JOIN + SUM(dotProduct) + re-chunk, ClickHouse dialect."""
    cs = str(chunk_size)
    dp = out_table + "_dp"

    client.command(f"DROP TEMPORARY TABLE IF EXISTS {dp}")
    client.command(f"DROP TEMPORARY TABLE IF EXISTS {out_table}")

    client.command(
        f"CREATE TEMPORARY TABLE {dp} ENGINE=Memory AS ("
        f"SELECT a.row_index AS act_row, w.row_index AS out_col, "
        f"SUM(dotProduct(a.v, w.v)) AS val "
        f"FROM {act_table} a JOIN {weight_table} w "
        f"ON a.chunk_index = w.chunk_index "
        f"GROUP BY a.row_index, w.row_index)"
    )

    client.command(
        f"CREATE TEMPORARY TABLE {out_table} ENGINE=Memory AS ("
        f"SELECT act_row AS row_index, "
        f"out_col - (out_col % {cs}) AS chunk_index, "
        f"{_order_then_agg('out_col', 'val')} AS v "
        f"FROM {dp} "
        f"GROUP BY act_row, out_col - (out_col % {cs}))"
    )


# ──────────────────────────────────────────────────────────────────────
# Pivoted MatMul (§4.3, ClickHouse dialect)
# ──────────────────────────────────────────────────────────────────────

def materialize_weight_pivots(client, weight_table: str,
                              n_chunks: int, chunk_size: int,
                              pivot_width: int) -> set[str]:
    """Pre-pivot the weight table for a given pivot_width (D9 cache).

    Mirrors ``ClickHouseRunner._pivot_weight_tables``. Returns the set of
    materialised pivot-table names so ``run_pivoted`` skips recreating
    them. Timed outside the measurement loop, matching
    ``pivot_setup_time_s`` accounting in run_clickhouse_prefill.
    """
    steps = pivoted_matmul_sql(
        "__dummy_act__", weight_table, "__dummy_out__",
        n_chunks, chunk_size, pivot_width, subquery_width=0,
    )

    cached: set[str] = set()
    for sql, name in steps:
        if name.startswith(f"{weight_table}_piv"):
            client.command(f"DROP TEMPORARY TABLE IF EXISTS {name}")
            client.command(
                f"CREATE TEMPORARY TABLE {name} ENGINE=Memory AS ({sql})"
            )
            cached.add(name)
    return cached


def run_pivoted(client, act_table: str, weight_table: str,
                out_table: str, n_chunks: int, chunk_size: int,
                pivot_width: int,
                cached_weight_pivots: set[str] | None = None) -> None:
    """Pivoted MatMul with §4.1 CTE merge.

    Equivalent to one ``postopt_dag_to_sql_ch`` group for a single MatMul
    with ``row2col_pivot=True``, ``cte_merge=True``. Cached weight
    pivots are stripped so we only time the non-cached portion, exactly
    as run_clickhouse_prefill does.
    """
    steps = pivoted_matmul_sql(
        act_table, weight_table, out_table,
        n_chunks, chunk_size, pivot_width, subquery_width=0,
    )

    cache = cached_weight_pivots or set()

    for _, name in steps:
        if name in cache:
            continue
        client.command(f"DROP TEMPORARY TABLE IF EXISTS {name}")

    non_cached = [(sql, name) for sql, name in steps if name not in cache]
    if not non_cached:
        return
    merged_sql, merged_name = _emit_cte_block(non_cached)
    client.command(
        f"CREATE TEMPORARY TABLE {merged_name} ENGINE=Memory AS ({merged_sql})"
    )


# ──────────────────────────────────────────────────────────────────────
# Timing
# ──────────────────────────────────────────────────────────────────────

def benchmark_one(client, act_table: str, weight_table: str,
                  n_chunks: int, chunk_size: int, pivot_width: int,
                  warmup: int = 1, repeats: int = 3,
                  cached_weight_pivots: set[str] | None = None
                  ) -> tuple[float, float]:
    for _ in range(warmup):
        run_pivoted(client, act_table, weight_table, "bench_out",
                    n_chunks, chunk_size, pivot_width,
                    cached_weight_pivots=cached_weight_pivots)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run_pivoted(client, act_table, weight_table, "bench_out",
                    n_chunks, chunk_size, pivot_width,
                    cached_weight_pivots=cached_weight_pivots)
        times.append(time.perf_counter() - start)

    return min(times), sum(times) / len(times)


def benchmark_baseline(client, act_table: str, weight_table: str,
                       chunk_size: int, warmup: int = 1,
                       repeats: int = 3) -> tuple[float, float]:
    for _ in range(warmup):
        run_baseline(client, act_table, weight_table,
                     "bench_base", chunk_size)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run_baseline(client, act_table, weight_table,
                     "bench_base", chunk_size)
        times.append(time.perf_counter() - start)

    return min(times), sum(times) / len(times)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune ROW2COL pivot_width on ClickHouse (§4.3)")
    parser.add_argument("--ch-host", default="localhost")
    parser.add_argument("--ch-port", type=int, default=8123)
    parser.add_argument("--ch-user", default="default")
    parser.add_argument("--ch-password", default="")
    parser.add_argument("--ch-database", default="default")
    parser.add_argument("--chunk-size", type=int, default=32,
                        help="Chunk size -- must match the stored weight "
                             "chunk_size (default: 32)")
    parser.add_argument("--max-threads", type=str, default="4",
                        help="Comma-separated ClickHouse max_threads to "
                             "test (default: 4 -- matches paper c7.2xlarge)")
    parser.add_argument("--lengths", type=str, default="25,50,100,200",
                        help="Comma-separated prompt lengths to tune for "
                             "(default matches run_clickhouse_prefill)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Timing repeats per config")
    parser.add_argument("--quick", action="store_true",
                        help="Test fewer configurations for faster results")
    # Paper's hardware: AWS c7.2xlarge = 4 cores, 16 GB RAM.
    parser.add_argument("--max-memory-usage", type=parse_bytes, default=None,
                        help="ClickHouse max_memory_usage. Accepts raw "
                             "bytes or a suffix (e.g. '16GB'). Paper: 16GB.")
    args = parser.parse_args()

    cs = args.chunk_size
    thread_list = [int(t) for t in args.max_threads.split(",")]
    lengths = [int(L) for L in args.lengths.split(",")]
    repeats = args.repeats

    # Llama3-8B projection shapes (paper §4.3). Realistic out_dim is
    # critical: CROSS JOIN cardinality scales with act_rows * weight_rows,
    # so a tiny weight out_dim hides the §4.3 speedup.
    matmul_shapes = [
        ("Q/O",     4096,  4096),
        ("K/V",     4096,  1024),
        ("gate/up", 4096, 14336),
        ("down",   14336,  4096),
    ]
    test_cases = []
    for L in lengths:
        for label, contracted, out_dim in matmul_shapes:
            name = (f"L={L} {label} (act {L}x{contracted}, "
                    f"wt {out_dim}x{contracted})")
            test_cases.append((name, L, contracted, out_dim))

    rng = np.random.default_rng(42)

    # Collect speedups keyed by pw across all matrices + threads to
    # compute a global "best on average" recommendation at the end.
    all_speedups: dict[int, list[float]] = {}

    default_opts = PostOptOptions()

    for name, m, contracted, out_dim in test_cases:
        assert contracted % cs == 0, (
            f"{name}: contracted={contracted} not divisible "
            f"by chunk_size={cs}")
        n_chunks = contracted // cs

        print(f"\n{'='*70}")
        print(f"Matrix: {name}")
        print(f"  chunk_size={cs}, n_chunks={n_chunks}")
        print(f"{'='*70}")

        # Same should_pivot heuristic the real runner applies.
        if not should_pivot(n_chunks, cs, default_opts):
            print(f"  SKIP: should_pivot heuristic rejects this matrix "
                  f"(n_chunks={n_chunks} > max_pivot_chunks="
                  f"{default_opts.max_pivot_chunks}). "
                  f"run_clickhouse_prefill runs this MatMul via the "
                  f"baseline path.")
            continue

        act_np = rng.standard_normal((m, contracted)).astype(np.float32) * 0.01
        wt_np = rng.standard_normal((out_dim, contracted)).astype(np.float32) * 0.01

        # Pivot widths to sweep: factors of n_chunks plus full-width.
        if args.quick:
            pivot_widths = [n_chunks]
        else:
            pivot_widths = sorted({
                pw for pw in [1, 2, 4, 8, 16, 32, 64, n_chunks]
                if pw <= n_chunks
            })

        for threads in thread_list:
            settings: dict[str, int] = {"max_threads": threads}
            if args.max_memory_usage is not None:
                settings["max_memory_usage"] = args.max_memory_usage

            # Fresh session per (matrix, thread) combo: temp tables die
            # with the session, so no manual cleanup needed between runs.
            session_id = f"tune_pw_{uuid.uuid4().hex[:12]}"
            client = clickhouse_connect.get_client(
                host=args.ch_host, port=args.ch_port,
                username=args.ch_user, password=args.ch_password,
                database=args.ch_database,
                session_id=session_id,
                settings=settings,
            )

            try:
                load_2d(client, "act", act_np, cs)
                load_2d(client, "wt", wt_np, cs)

                base_min, base_avg = benchmark_baseline(
                    client, "act", "wt", cs, repeats=repeats)
                print(f"\n  max_threads={threads}, Baseline: "
                      f"min={base_min:.3f}s, avg={base_avg:.3f}s")
                print(f"  {'pivot_width':>12} {'min_time':>10} "
                      f"{'avg_time':>10} {'speedup':>8}")
                print(f"  {'-'*45}")

                best_speedup = 0.0
                best_pw = 0

                for pw in pivot_widths:
                    try:
                        cached = materialize_weight_pivots(
                            client, "wt", n_chunks, cs, pw)
                    except Exception as e:
                        print(f"  {pw:>12}  CACHE ERROR: {e}")
                        continue

                    try:
                        piv_min, piv_avg = benchmark_one(
                            client, "act", "wt", n_chunks, cs, pw,
                            repeats=repeats, cached_weight_pivots=cached)
                        speedup = base_avg / piv_avg if piv_avg > 0 else 0
                        print(f"  {pw:>12} {piv_min:>10.3f} "
                              f"{piv_avg:>10.3f} {speedup:>7.2f}x")

                        if speedup > best_speedup:
                            best_speedup = speedup
                            best_pw = pw
                        all_speedups.setdefault(pw, []).append(speedup)
                    except Exception as e:
                        print(f"  {pw:>12}  ERROR: {e}")

                    # Drop cached pivots before next pw to free memory.
                    for piv_name in cached:
                        client.command(
                            f"DROP TEMPORARY TABLE IF EXISTS {piv_name}")

                print(f"\n  Best: pivot_width={best_pw}, "
                      f"speedup={best_speedup:.2f}x")
            finally:
                client.close()

    # ── Global recommendation ──
    # A single pivot_width is passed to run_clickhouse_prefill for the
    # full benchmark, so we want the value with the highest geometric-mean
    # speedup across every tested (matrix, threads) combination.
    if all_speedups:
        print(f"\n{'='*70}")
        print(f"Global recommendation (geometric-mean speedup across all "
              f"matrices + threads)")
        print(f"{'='*70}")
        scored = []
        for pw, speedups in all_speedups.items():
            if not speedups:
                continue
            geomean = math.exp(
                sum(math.log(s) for s in speedups if s > 0)
                / len(speedups)
            )
            scored.append((pw, geomean, min(speedups), max(speedups)))
        scored.sort(key=lambda x: x[1], reverse=True)

        print(f"  {'pivot_width':>12} {'geomean':>10} "
              f"{'min':>8} {'max':>8}")
        print(f"  {'-'*45}")
        for pw, gm, mn, mx in scored[:10]:
            print(f"  {pw:>12} {gm:>9.2f}x {mn:>7.2f}x {mx:>7.2f}x")
        best_pw = scored[0][0]
        print(f"\nSuggested flag for run_clickhouse_prefill / "
              f"run_clickhouse_decode / run_clickhouse_perplexity:")
        print(f"  --pivot-width {best_pw}")


if __name__ == "__main__":
    main()
