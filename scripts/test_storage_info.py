"""Inspect what compression DuckDB applies to FLOAT[32] array columns.

If columns are stored 'Uncompressed', we can try forcing Chimp/Patas.
"""
import duckdb

con = duckdb.connect("weights_checkpoint_test.duckdb", read_only=True)

tables = [r[0] for r in con.execute(
    "SELECT name FROM (SHOW TABLES) LIMIT 3"
).fetchall()]

for t in tables:
    print(f"\n=== {t} ===")
    rows = con.execute(f"PRAGMA storage_info('{t}')").fetchall()
    cols = [d[0] for d in con.description]
    print("  columns:", cols)
    compressions = {}
    for r in rows:
        rec = dict(zip(cols, r))
        col = rec.get("column_name") or rec.get("column_path") or rec.get("column_id")
        comp = rec.get("compression") or rec.get("segment_type")
        key = (col, comp)
        compressions[key] = compressions.get(key, 0) + 1
    for (col, comp), n in sorted(compressions.items()):
        print(f"  col={col} compression={comp} segments={n}")

con.close()

print("\n=== Try forcing Chimp compression ===")
import os, shutil
if os.path.exists("weights_forced_chimp.duckdb"):
    os.remove("weights_forced_chimp.duckdb")

con = duckdb.connect("weights_forced_chimp.duckdb")
con.execute("SET force_compression='chimp'")
con.execute("ATTACH 'weights_checkpoint_test.duckdb' AS src (READ_ONLY)")
t = tables[0]
con.execute(f"CREATE TABLE {t} AS SELECT * FROM src.{t}")
con.execute("DETACH src")
con.execute("FORCE CHECKPOINT")
con.close()

size_default = os.path.getsize("weights_checkpoint_test.duckdb") / 1024**3
size_chimp = os.path.getsize("weights_forced_chimp.duckdb") / 1024**3
print(f"default compression: {size_default:.3f} GB")
print(f"force_compression=chimp: {size_chimp:.3f} GB")
print(f"Note: chimp db only contains 1 table, default db contains 9 tables - adjust mentally")
