"""
Load chunked CSV weight files into Umbra (PostgreSQL wire protocol).

Mirrors the DuckDB loader's table layout, substituting PostgreSQL types:

    2D weight: (row_index INTEGER, chunk_index INTEGER, v REAL[])
    1D norm:   (chunk_index INTEGER, v REAL[])   (Decision D1)
    RoPE:      (row_index INTEGER, chunk_index INTEGER,
                cos REAL[], sin REAL[])

Connects via psycopg2 (port 15432, user postgres).

Umbra's PostgreSQL compatibility has some quirks with psycopg2's
automatic type adaptation, so we build VALUES literals manually
using PostgreSQL array syntax ('{1.0, 2.0, ...}').

Usage:
    python -m preprocessing.load_weights_umbra \\
        --csv-dir weights_csv \\
        [--host localhost] [--port 15432] \\
        [--chunk-size 32] [--num-layers 32]
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import os
import time

import numpy as np
import psycopg2
import psycopg2.extensions

from transql_plus.config import ModelConfig


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _is_norm_table(name: str) -> bool:
    return name.endswith("_norm1") or name.endswith("_norm2") or name == "final_norm"


def _is_moe_table(name: str) -> bool:
    return ("_moe_gate_proj" in name or "_moe_up_proj" in name
            or "_moe_down_proj" in name)


def table_schema(name: str) -> str:
    """Return CREATE TABLE DDL in PostgreSQL dialect."""
    if name == "rope":
        return (
            "CREATE TABLE rope ("
            "row_index INTEGER NOT NULL, "
            "chunk_index INTEGER NOT NULL, "
            "cos REAL[], sin REAL[], "
            "PRIMARY KEY (row_index, chunk_index))"
        )

    if _is_norm_table(name):
        return (
            f"CREATE TABLE {name} ("
            f"chunk_index INTEGER NOT NULL PRIMARY KEY, "
            f"v REAL[])"
        )

    if _is_moe_table(name):
        return (
            f"CREATE TABLE {name} ("
            f"expert_id INTEGER NOT NULL, "
            f"row_index INTEGER NOT NULL, "
            f"chunk_index INTEGER NOT NULL, "
            f"v REAL[], "
            f"PRIMARY KEY (expert_id, row_index, chunk_index))"
        )

    return (
        f"CREATE TABLE {name} ("
        f"row_index INTEGER NOT NULL, "
        f"chunk_index INTEGER NOT NULL, "
        f"v REAL[], "
        f"PRIMARY KEY (row_index, chunk_index))"
    )


def _pg_array(vals: list[float]) -> str:
    """Format a Python list as a PostgreSQL array literal: '{1.0, 2.0}'."""
    inner = ", ".join(f"{v}" for v in vals)
    return f"'{{{inner}}}'"


# ---------------------------------------------------------------------------
# Filtering / selection
# ---------------------------------------------------------------------------

def _select_csvs(csv_dir: str, num_layers: int) -> list[str]:
    """Return CSVs belonging to the first ``num_layers`` layers + globals."""
    all_csvs = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    kept: list[str] = []
    for path in all_csvs:
        name = os.path.splitext(os.path.basename(path))[0]
        if name.startswith("layer_"):
            layer_id = int(name.split("_")[1])
            if layer_id < num_layers:
                kept.append(path)
        else:
            kept.append(path)
    return kept


# ---------------------------------------------------------------------------
# RoPE from formula
# ---------------------------------------------------------------------------

def load_rope_from_formula(con, config: ModelConfig) -> None:
    """Compute and load the rope table directly from the RoPE formula."""
    cs = config.chunk_size
    half = cs // 2
    num_chunks = config.hidden_dim // cs
    head_dim = config.head_dim
    rope_theta = config.rope_theta
    max_seq = config.max_seq_len

    pos = np.repeat(np.arange(max_seq, dtype=np.int32), num_chunks * half)
    chunk = np.tile(
        np.repeat(np.arange(num_chunks, dtype=np.int32), half), max_seq,
    )
    pair = np.tile(np.arange(half, dtype=np.int32), max_seq * num_chunks)
    chunk_index = chunk.astype(np.int32) * cs

    global_dim = chunk.astype(np.int32) * cs + 2 * pair
    d_in_head = global_dim % head_dim
    pair_idx = d_in_head // 2
    inv_freq = (1.0 / (rope_theta ** (2.0 * pair_idx / head_dim))
                ).astype(np.float64)
    angle = pos.astype(np.float64) * inv_freq
    cos_vals = np.cos(angle).astype(np.float32)
    sin_vals = np.sin(angle).astype(np.float32)

    # Group into (row_index, chunk_index, cos[], sin[]) rows
    rows: dict[tuple[int, int], tuple[list[float], list[float]]] = {}
    for i in range(len(pos)):
        key = (int(pos[i]), int(chunk_index[i]))
        if key not in rows:
            rows[key] = ([], [])
        rows[key][0].append(float(cos_vals[i]))
        rows[key][1].append(float(sin_vals[i]))

    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS rope")
    cur.execute(table_schema("rope"))

    batch_size = 500
    batch_parts: list[str] = []
    inserted = 0
    for (ri, ci), (cos_list, sin_list) in sorted(rows.items()):
        batch_parts.append(
            f"({ri}, {ci}, {_pg_array(cos_list)}, {_pg_array(sin_list)})"
        )
        if len(batch_parts) >= batch_size:
            cur.execute(
                "INSERT INTO rope (row_index, chunk_index, cos, sin) "
                "VALUES " + ", ".join(batch_parts)
            )
            inserted += len(batch_parts)
            batch_parts.clear()
    if batch_parts:
        cur.execute(
            "INSERT INTO rope (row_index, chunk_index, cos, sin) "
            "VALUES " + ", ".join(batch_parts)
        )
        inserted += len(batch_parts)
    con.commit()

    cur.execute("SELECT COUNT(*) FROM rope")
    cnt = cur.fetchone()[0]
    print(f"  rope: {cnt} rows (computed from formula, theta={rope_theta})")
    cur.close()


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_table_from_csv(con, csv_path: str, table_name: str) -> None:
    """Load a single CSV file into an Umbra table."""
    cur = con.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {table_name}")
    cur.execute(table_schema(table_name))

    t0 = time.perf_counter()
    batch_size = 500

    is_norm = _is_norm_table(table_name)
    is_moe = _is_moe_table(table_name)

    batch_parts: list[str] = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vals = ast.literal_eval(row["v"])
            arr = _pg_array(vals)

            if is_norm:
                batch_parts.append(f"({int(row['chunk_index'])}, {arr})")
            elif is_moe:
                batch_parts.append(
                    f"({int(row['expert_id'])}, {int(row['row_index'])}, "
                    f"{int(row['chunk_index'])}, {arr})"
                )
            else:
                batch_parts.append(
                    f"({int(row['row_index'])}, {int(row['chunk_index'])}, "
                    f"{arr})"
                )

            if len(batch_parts) >= batch_size:
                _flush_sql(cur, table_name, batch_parts, is_norm, is_moe)
                batch_parts.clear()

    if batch_parts:
        _flush_sql(cur, table_name, batch_parts, is_norm, is_moe)
    con.commit()

    dt = time.perf_counter() - t0
    size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    cnt = cur.fetchone()[0]
    print(f"  {table_name}: {cnt} rows  "
          f"({size_mb:.1f} MB in {dt:.2f}s)")
    cur.close()


def _flush_sql(cur, table_name: str, parts: list[str],
               is_norm: bool, is_moe: bool) -> None:
    """Execute a multi-row INSERT with pre-formatted VALUES strings."""
    if is_norm:
        cols = "(chunk_index, v)"
    elif is_moe:
        cols = "(expert_id, row_index, chunk_index, v)"
    else:
        cols = "(row_index, chunk_index, v)"
    cur.execute(
        f"INSERT INTO {table_name} {cols} VALUES " + ", ".join(parts)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_all(csv_dir: str, config: ModelConfig, *,
             host: str, port: int, user: str, password: str,
             num_layers: int) -> None:
    print(f"Connecting to Umbra at {host}:{port}...")
    con = psycopg2.connect(
        host=host, port=port, user=user, password=password,
    )
    # Use manual commits for batch inserts
    con.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_READ_COMMITTED)

    csv_files = _select_csvs(csv_dir, num_layers)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matched under {csv_dir}")

    print(f"Computing rope table (theta={config.rope_theta})...")
    load_rope_from_formula(con, config)

    non_rope = [f for f in csv_files
                if os.path.splitext(os.path.basename(f))[0] != "rope"]
    print(f"Loading {len(non_rope)} weight tables from CSV "
          f"(num_layers={num_layers})...")

    t_all = time.perf_counter()
    for csv_path in non_rope:
        table_name = os.path.splitext(os.path.basename(csv_path))[0]
        load_table_from_csv(con, csv_path, table_name)
    total = time.perf_counter() - t_all

    con.close()
    print(f"\nDone. {len(non_rope) + 1} tables loaded into Umbra "
          f"in {total:.1f}s.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load chunked CSV weights into Umbra (PostgreSQL)")
    parser.add_argument("--csv-dir", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=15432)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="umbra")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=32,
                        help="Number of transformer layers to load "
                             "(smoke tests can use 1).")
    parser.add_argument("--rope-theta", type=float, default=500000.0)
    args = parser.parse_args()

    config = ModelConfig.llama3_8b(chunk_size=args.chunk_size)
    config.rope_theta = args.rope_theta

    load_all(args.csv_dir, config,
             host=args.host, port=args.port,
             user=args.user, password=args.password,
             num_layers=args.num_layers)


if __name__ == "__main__":
    main()
