"""
Fast benchmark: does caching the weight PIVOT help per-call latency?

Compares per-call latency of one MatMul at Llama3-8B hidden dims
(4096 hidden, chunk_size=32, 128 chunks, seq_len=25):

  A) Inline:  PIVOT weight inside each query (re-computed every call)
  B) Cached:  CREATE TEMP TABLE weight_piv once, reuse across calls

Reports mean +/- std per variant and the break-even point (#runs at
which the one-time pivot setup pays for itself).

Usage:
    python scripts/bench_pivot_cache.py
"""
from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import duckdb

from transql_plus.postopt import pivot_sql, _chunk_offsets


def _build_random_chunked_table(con, name: str, n_rows: int,
                                n_chunks: int, chunk_size: int) -> None:
    """Materialize (row_index, chunk_index, v FLOAT[cs]) with random data."""
    con.execute(f"""
        CREATE TABLE {name} AS
        SELECT r AS row_index, c * {chunk_size} AS chunk_index,
               CAST(list_transform(generate_series(1, {chunk_size}),
                                   x -> CAST(random() - 0.5 AS FLOAT))
                    AS FLOAT[{chunk_size}]) AS v
        FROM generate_series(0, {n_rows - 1}) t1(r),
             generate_series(0, {n_chunks - 1}) t2(c)
    """)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="16GB")
    args = parser.parse_args()

    hidden = args.hidden
    cs = args.chunk_size
    n_chunks = hidden // cs
    seq_len = args.seq_len

    tmpdir = Path(tempfile.mkdtemp(prefix="bench_pivot_"))
    db_path = str(tmpdir / "bench.duckdb")
    temp_dir = str(tmpdir / "duckdb_tmp")
    (tmpdir / "duckdb_tmp").mkdir(exist_ok=True)

    con = duckdb.connect(db_path, config={
        "threads": args.threads,
        "memory_limit": args.memory_limit,
        "temp_directory": temp_dir,
    })

    print(f"Config: hidden={hidden}, chunk_size={cs}, n_chunks={n_chunks}, "
          f"seq_len={seq_len}, threads={args.threads}, "
          f"memory_limit={args.memory_limit}")

    # Synthetic weight (hidden x hidden) and activation (seq_len x hidden)
    print("Building synthetic weight/activation tables...")
    t0 = time.perf_counter()
    _build_random_chunked_table(con, "weight_tbl", hidden, n_chunks, cs)
    _build_random_chunked_table(con, "act_tbl", seq_len, n_chunks, cs)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    offsets = _chunk_offsets(0, n_chunks, cs)
    dot_cols = [f'"{o}"' for o in offsets]
    dot_expr = " + ".join(f"list_dot_product(a.{c}, w.{c})" for c in dot_cols)

    # Pre-pivot activation (constant across variants — isolates weight pivot cost)
    con.execute(
        f"CREATE TEMP TABLE act_piv AS ({pivot_sql('act_tbl', offsets)})"
    )

    # ── Variant A: inline weight pivot (mimics CTE-merged query) ──
    inline_sql = f"""
    WITH w_piv AS ({pivot_sql('weight_tbl', offsets)})
    SELECT COUNT(*) FROM (
      SELECT a.row_index AS act_row, w.row_index AS out_col,
             {dot_expr} AS val
      FROM act_piv a CROSS JOIN w_piv w
    )
    """

    print(f"\n[A] Inline weight pivot: {args.warmup} warmup + "
          f"{args.repeat} measured")
    for i in range(args.warmup):
        t0 = time.perf_counter()
        con.execute(inline_sql).fetchone()
        print(f"  warmup {i+1}: {time.perf_counter()-t0:.3f}s")
    inline_times = []
    for i in range(args.repeat):
        t0 = time.perf_counter()
        con.execute(inline_sql).fetchone()
        dt = time.perf_counter() - t0
        inline_times.append(dt)
        print(f"  run    {i+1}: {dt:.3f}s")

    # ── Variant B: cached weight pivot ──
    print(f"\n[B] Cached weight pivot: materialize once, reuse")
    t0 = time.perf_counter()
    con.execute(
        f"CREATE TEMP TABLE weight_piv_cached AS "
        f"({pivot_sql('weight_tbl', offsets)})"
    )
    pivot_setup = time.perf_counter() - t0
    print(f"  pivot setup (once): {pivot_setup:.3f}s")

    cached_sql = f"""
    SELECT COUNT(*) FROM (
      SELECT a.row_index AS act_row, w.row_index AS out_col,
             {dot_expr} AS val
      FROM act_piv a CROSS JOIN weight_piv_cached w
    )
    """

    print(f"  {args.warmup} warmup + {args.repeat} measured")
    for i in range(args.warmup):
        t0 = time.perf_counter()
        con.execute(cached_sql).fetchone()
        print(f"  warmup {i+1}: {time.perf_counter()-t0:.3f}s")
    cached_times = []
    for i in range(args.repeat):
        t0 = time.perf_counter()
        con.execute(cached_sql).fetchone()
        dt = time.perf_counter() - t0
        cached_times.append(dt)
        print(f"  run    {i+1}: {dt:.3f}s")

    # ── Summary ──
    inline_mean = statistics.mean(inline_times)
    inline_std = statistics.stdev(inline_times) if len(inline_times) > 1 else 0.0
    cached_mean = statistics.mean(cached_times)
    cached_std = statistics.stdev(cached_times) if len(cached_times) > 1 else 0.0

    print(f"\n─── Summary ───")
    print(f"  Pivot setup (one-time): {pivot_setup:.3f}s")
    print(f"  [A] Inline  per-run:    {inline_mean:.3f}s  "
          f"(+/- {inline_std:.3f})")
    print(f"  [B] Cached  per-run:    {cached_mean:.3f}s  "
          f"(+/- {cached_std:.3f})")

    savings = inline_mean - cached_mean
    if savings > 0:
        breakeven = pivot_setup / savings
        print(f"  Per-run savings:        {savings:.3f}s "
              f"({100*savings/inline_mean:.1f}%)")
        print(f"  Break-even at:          {breakeven:.2f} runs")
    else:
        print(f"  Cached is NOT faster (diff = {-savings:.3f}s). "
              f"Caching has no benefit here.")

    # ── Correctness: A and B must produce identical dot products ──
    # Both run on the same weight_tbl / act_tbl, so results should be
    # bitwise-equal (PIVOT is deterministic; float summation order inside
    # list_dot_product is fixed). Compare per-cell and report max |diff|.
    print(f"\n─── Correctness check ───")
    inline_full_sql = f"""
    WITH w_piv AS ({pivot_sql('weight_tbl', offsets)})
    SELECT a.row_index AS act_row, w.row_index AS out_col,
           {dot_expr} AS val
    FROM act_piv a CROSS JOIN w_piv w
    ORDER BY act_row, out_col
    """
    cached_full_sql = f"""
    SELECT a.row_index AS act_row, w.row_index AS out_col,
           {dot_expr} AS val
    FROM act_piv a CROSS JOIN weight_piv_cached w
    ORDER BY act_row, out_col
    """
    rows_a = con.execute(inline_full_sql).fetchall()
    rows_b = con.execute(cached_full_sql).fetchall()

    print(f"  Rows A: {len(rows_a)}   Rows B: {len(rows_b)}")
    assert len(rows_a) == len(rows_b), "row count mismatch"

    max_abs_diff = 0.0
    mismatches = 0
    for (r1, c1, v1), (r2, c2, v2) in zip(rows_a, rows_b):
        if r1 != r2 or c1 != c2:
            mismatches += 1
            continue
        diff = abs(v1 - v2)
        if diff > max_abs_diff:
            max_abs_diff = diff
    print(f"  Key mismatches: {mismatches}")
    print(f"  Max |A - B|:    {max_abs_diff:.6e}")
    if mismatches == 0 and max_abs_diff == 0.0:
        print(f"  PASS: A and B produce bitwise-identical results.")
    elif mismatches == 0 and max_abs_diff < 1e-5:
        print(f"  PASS: A and B agree within float tolerance.")
    else:
        print(f"  FAIL: results diverge.")

    con.close()


if __name__ == "__main__":
    main()
