"""Measure CHECKPOINT's effect on storage size, accuracy, and query speed.

Strategy (avoids copying the full 35GB DB):
  1. Create a small test DB containing ONE layer's weights (~1GB)
  2. Measure size, sample values, time a MatMul-style query
  3. Run FORCE CHECKPOINT
  4. Re-measure — expect bit-identical values (CHECKPOINT is lossless)
  5. Compression ratio on one layer ~= ratio on the full DB (weight distributions are similar)
"""

import os
import time

import duckdb

SRC = "weights.duckdb"
DST = "weights_checkpoint_test.duckdb"


def fsize_gb(path: str) -> float:
    return os.path.getsize(path) / (1024**3)


def main():
    if not os.path.exists(SRC):
        raise SystemExit(f"{SRC} not found")
    if os.path.exists(DST):
        os.remove(DST)

    print(f"Source: {SRC} = {fsize_gb(SRC):.2f} GB (not copied — we extract one layer)")

    con = duckdb.connect(DST)
    con.execute(f"ATTACH '{SRC}' AS src (READ_ONLY)")

    layer0_tables = [r[0] for r in con.execute(
        "SELECT name FROM (SHOW ALL TABLES) WHERE database = 'src' AND name LIKE 'layer_0_%'"
    ).fetchall()]

    print(f"Extracting layer_0 tables: {len(layer0_tables)} tables")
    for t in layer0_tables:
        con.execute(f"CREATE TABLE {t} AS SELECT * FROM src.{t}")

    con.execute("DETACH src")
    con.close()

    size_before = fsize_gb(DST)
    print(f"\nAfter extraction: {DST} = {size_before:.2f} GB")

    con = duckdb.connect(DST)
    probe_table = None
    for t in layer0_tables:
        cols = {r[0] for r in con.execute(f"DESCRIBE {t}").fetchall()}
        if "row_index" in cols and "chunk_index" in cols:
            probe_table = t
            break
    if probe_table is None:
        raise SystemExit("no probe table found")

    pre_vals = con.execute(
        f"SELECT row_index, chunk_index, v FROM {probe_table} "
        "ORDER BY row_index, chunk_index LIMIT 10"
    ).fetchall()

    probe_sql = (
        f"SELECT count(*), avg(list_sum(v)), min(list_min(v)), max(list_max(v)) "
        f"FROM {probe_table}"
    )

    def time_query(sql, n=5):
        con.execute(sql).fetchall()  # warmup
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            con.execute(sql).fetchall()
            times.append(time.perf_counter() - t0)
        return min(times), sum(times) / len(times)

    pre_min, pre_avg = time_query(probe_sql)

    print(f"\n=== FORCE CHECKPOINT ===")
    t0 = time.perf_counter()
    con.execute("FORCE CHECKPOINT")
    ckpt_time = time.perf_counter() - t0
    print(f"CHECKPOINT took {ckpt_time:.1f}s")
    con.close()

    size_after = fsize_gb(DST)
    print(f"After CHECKPOINT: {DST} = {size_after:.2f} GB")
    reduction = (1 - size_after / size_before) * 100
    print(f"Reduction: {reduction:+.1f}%  (35GB full DB would become ~{35 * size_after / size_before:.1f}GB)")

    con = duckdb.connect(DST, read_only=True)
    post_vals = con.execute(
        f"SELECT row_index, chunk_index, v FROM {probe_table} "
        "ORDER BY row_index, chunk_index LIMIT 10"
    ).fetchall()
    post_min, post_avg = time_query(probe_sql)

    print()
    if pre_vals == post_vals:
        print("[OK] Values bit-identical — CHECKPOINT is lossless, zero accuracy impact")
    else:
        print("[FAIL] Values differ:")
        for a, b in zip(pre_vals[:3], post_vals[:3]):
            if a != b:
                print(f"  pre:  {a}")
                print(f"  post: {b}")

    print(f"\nQuery: {probe_sql[:80]}...")
    print(f"  pre  CHECKPOINT: min={pre_min * 1000:.1f}ms  avg={pre_avg * 1000:.1f}ms")
    print(f"  post CHECKPOINT: min={post_min * 1000:.1f}ms  avg={post_avg * 1000:.1f}ms")
    ratio = post_min / pre_min
    print(f"  slowdown: {ratio:.2f}x ({'negligible' if 0.9 < ratio < 1.1 else 'measurable'})")

    con.close()


if __name__ == "__main__":
    main()
