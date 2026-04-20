"""
Load chunked CSV weight files into ClickHouse (companion to
``preprocessing/load_weights_duckdb.py``).

Mirrors the DuckDB loader's table layout, substituting ClickHouse types:

    2D weight: (row_index Int32, chunk_index Int32, v Array(Float32))
               ENGINE=MergeTree() ORDER BY (row_index, chunk_index)
    1D norm:   (chunk_index Int32, v Array(Float32))   (Decision D1)
               ENGINE=MergeTree() ORDER BY chunk_index
    RoPE:      (row_index Int32, chunk_index Int32,
                cos Array(Float32), sin Array(Float32))
               ENGINE=MergeTree() ORDER BY (row_index, chunk_index)

Inserts use ``raw_insert`` with format ``CSVWithNames`` so the CSV is
streamed to the server without Python-side parsing (the same files the
DuckDB loader consumes).

Usage:
    python -m preprocessing.load_weights_clickhouse \\
        --csv-dir weights_csv --ch-host localhost --ch-port 8123 \\
        --ch-database default [--chunk-size 32] [--num-layers 32]
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import clickhouse_connect
import numpy as np
import pandas as pd

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
    """Return ``CREATE TABLE`` DDL for a weight table in ClickHouse dialect."""
    if name == "rope":
        return (
            "CREATE TABLE rope ("
            "row_index Int32, chunk_index Int32, "
            "cos Array(Float32), sin Array(Float32)"
            ") ENGINE=MergeTree() ORDER BY (row_index, chunk_index)"
        )

    if _is_norm_table(name):
        return (
            f"CREATE TABLE {name} ("
            f"chunk_index Int32, v Array(Float32)"
            f") ENGINE=MergeTree() ORDER BY chunk_index"
        )

    if _is_moe_table(name):
        return (
            f"CREATE TABLE {name} ("
            f"expert_id Int32, row_index Int32, "
            f"chunk_index Int32, v Array(Float32)"
            f") ENGINE=MergeTree() ORDER BY (expert_id, row_index, chunk_index)"
        )

    return (
        f"CREATE TABLE {name} ("
        f"row_index Int32, chunk_index Int32, v Array(Float32)"
        f") ENGINE=MergeTree() ORDER BY (row_index, chunk_index)"
    )


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
# RoPE from formula (bypasses CSV for precision — same approach as DuckDB)
# ---------------------------------------------------------------------------

def load_rope_from_formula(client, config: ModelConfig) -> None:
    cs = config.chunk_size
    half = cs // 2
    num_chunks = config.hidden_dim // cs
    head_dim = config.head_dim
    rope_theta = config.rope_theta
    max_seq = config.max_seq_len

    pos = np.repeat(np.arange(max_seq, dtype=np.int32),
                    num_chunks * half)
    chunk = np.tile(
        np.repeat(np.arange(num_chunks, dtype=np.int32), half), max_seq,
    )
    pair = np.tile(np.arange(half, dtype=np.int32),
                   max_seq * num_chunks)
    chunk_index = chunk.astype(np.int32) * cs

    global_dim = chunk.astype(np.int32) * cs + 2 * pair
    d_in_head = global_dim % head_dim
    pair_idx = d_in_head // 2
    inv_freq = (1.0 / (rope_theta ** (2.0 * pair_idx / head_dim))
                ).astype(np.float64)
    angle = pos.astype(np.float64) * inv_freq
    cos_vals = np.cos(angle).astype(np.float32)
    sin_vals = np.sin(angle).astype(np.float32)

    flat = pd.DataFrame({
        "pos": pos,
        "chunk_index": chunk_index,
        "pair": pair,
        "cos_val": cos_vals,
        "sin_val": sin_vals,
    })

    # Group in pandas to build the (row_index, chunk_index, cos[], sin[]) rows.
    grouped = (
        flat.sort_values(["pos", "chunk_index", "pair"])
            .groupby(["pos", "chunk_index"], sort=False)
            .agg(cos=("cos_val", list), sin=("sin_val", list))
            .reset_index()
            .rename(columns={"pos": "row_index"})
    )

    client.command("DROP TABLE IF EXISTS rope")
    client.command(table_schema("rope"))
    client.insert_df("rope", grouped)
    cnt = client.query("SELECT count() FROM rope").result_rows[0][0]
    print(f"  rope: {cnt} rows (computed from formula, theta={rope_theta})")


# ---------------------------------------------------------------------------
# CSV loading — stream file to ClickHouse via raw_insert
# ---------------------------------------------------------------------------

def load_table_from_csv(client, csv_path: str,
                        table_name: str) -> None:
    client.command(f"DROP TABLE IF EXISTS {table_name}")
    client.command(table_schema(table_name))

    # CSVWithNames: the header row names the columns; ClickHouse matches
    # them to the destination columns (order-independent). Streams bytes
    # directly from disk so peak Python memory stays constant regardless
    # of table size.
    t0 = time.perf_counter()
    with open(csv_path, "rb") as f:
        client.raw_insert(
            table_name,
            insert_block=f,
            fmt="CSVWithNames",
        )
    dt = time.perf_counter() - t0
    size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    cnt = client.query(f"SELECT count() FROM {table_name}").result_rows[0][0]
    print(f"  {table_name}: {cnt} rows  "
          f"({size_mb:.1f} MB in {dt:.2f}s = "
          f"{size_mb / max(dt, 1e-6):.1f} MB/s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_all(csv_dir: str, config: ModelConfig, *,
             host: str, port: int, user: str, password: str,
             database: str, num_layers: int) -> None:
    print(f"Connecting to ClickHouse at {host}:{port} (db={database})...")
    client = clickhouse_connect.get_client(
        host=host, port=port, username=user, password=password,
        database=database,
    )

    csv_files = _select_csvs(csv_dir, num_layers)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matched under {csv_dir}")

    print(f"Computing rope table (theta={config.rope_theta})...")
    load_rope_from_formula(client, config)

    non_rope = [f for f in csv_files
                if os.path.splitext(os.path.basename(f))[0] != "rope"]
    print(f"Loading {len(non_rope)} weight tables from CSV "
          f"(num_layers={num_layers})...")

    t_all = time.perf_counter()
    for csv_path in non_rope:
        table_name = os.path.splitext(os.path.basename(csv_path))[0]
        load_table_from_csv(client, csv_path, table_name)
    total = time.perf_counter() - t_all

    print(f"\nDone. {len(non_rope) + 1} tables loaded into ClickHouse "
          f"({database}) in {total:.1f}s.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", required=True)
    parser.add_argument("--ch-host", default="localhost")
    parser.add_argument("--ch-port", type=int, default=8123)
    parser.add_argument("--ch-user", default="default")
    parser.add_argument("--ch-password", default="")
    parser.add_argument("--ch-database", default="default")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=32,
                        help="Number of transformer layers to load "
                             "(smoke tests can use 1).")
    parser.add_argument("--rope-theta", type=float, default=500000.0)
    args = parser.parse_args()

    config = ModelConfig.llama3_8b(chunk_size=args.chunk_size)
    config.rope_theta = args.rope_theta

    load_all(args.csv_dir, config,
             host=args.ch_host, port=args.ch_port,
             user=args.ch_user, password=args.ch_password,
             database=args.ch_database,
             num_layers=args.num_layers)


if __name__ == "__main__":
    main()
