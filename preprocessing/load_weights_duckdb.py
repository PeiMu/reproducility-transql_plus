"""
Load chunked CSV weight files into a DuckDB database.

Paper reference: Section 3.1.1.

Table schemas (matching sql_templates.py conventions):
    2D weight: (row_index INTEGER, chunk_index INTEGER, v FLOAT[cs],
                PRIMARY KEY (row_index, chunk_index))
    1D norm:   (chunk_index INTEGER PRIMARY KEY, v FLOAT[cs])  (Decision D1)
    RoPE:      (row_index INTEGER, chunk_index INTEGER,
                cos FLOAT[half], sin FLOAT[half],
                PRIMARY KEY (row_index, chunk_index))

Indexes:
    - chunk_index on all weight tables (MatMul JOIN performance)
    - row_index on embed_tokens (embedding lookup)
    - expert_id on MOE tables (§3.2)

Usage:
    python -m preprocessing.load_weights_duckdb \\
        --csv-dir weights_csv --db-path weights.duckdb [--chunk-size 32]
"""

from __future__ import annotations

import argparse
import glob
import os

import duckdb
import numpy as np
import pandas as pd

from transql_plus.config import ModelConfig


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _is_norm_table(name: str) -> bool:
    return name.endswith("_norm1") or name.endswith("_norm2") or name == "final_norm"


def _is_moe_table(name: str) -> bool:
    return "_moe_gate_proj" in name or "_moe_up_proj" in name or "_moe_down_proj" in name


def table_schema(name: str, chunk_size: int) -> str:
    """Return CREATE TABLE DDL for a given weight table."""
    cs = chunk_size
    half = cs // 2

    if name == "rope":
        return (
            f"CREATE TABLE rope ("
            f"row_index INTEGER NOT NULL, "
            f"chunk_index INTEGER NOT NULL, "
            f"cos FLOAT[{half}], sin FLOAT[{half}], "
            f"PRIMARY KEY (row_index, chunk_index))"
        )

    if _is_norm_table(name):
        # Decision D1: no row_index for 1D norms
        return (
            f"CREATE TABLE {name} ("
            f"chunk_index INTEGER NOT NULL PRIMARY KEY, "
            f"v FLOAT[{cs}])"
        )

    if _is_moe_table(name):
        return (
            f"CREATE TABLE {name} ("
            f"expert_id INTEGER NOT NULL, "
            f"row_index INTEGER NOT NULL, "
            f"chunk_index INTEGER NOT NULL, "
            f"v FLOAT[{cs}], "
            f"PRIMARY KEY (expert_id, row_index, chunk_index))"
        )

    # 2D weight or embedding
    return (
        f"CREATE TABLE {name} ("
        f"row_index INTEGER NOT NULL, "
        f"chunk_index INTEGER NOT NULL, "
        f"v FLOAT[{cs}], "
        f"PRIMARY KEY (row_index, chunk_index))"
    )


# ---------------------------------------------------------------------------
# RoPE from formula (bypasses CSV for precision)
# ---------------------------------------------------------------------------

def load_rope_from_formula(
    con: duckdb.DuckDBPyConnection,
    config: ModelConfig,
) -> None:
    """Compute and load the rope table directly from the RoPE formula.

    This guarantees the correct theta base regardless of how the CSV was
    generated. Produces raw offset chunk_index (Decision D7).
    """
    cs = config.chunk_size
    half = cs // 2
    num_chunks = config.hidden_dim // cs
    head_dim = config.head_dim
    rope_theta = config.rope_theta
    max_seq = config.max_seq_len

    # Vectorised computation
    pos = np.repeat(np.arange(max_seq, dtype=np.int32), num_chunks * half)
    chunk = np.tile(
        np.repeat(np.arange(num_chunks, dtype=np.int32), half), max_seq,
    )
    pair = np.tile(np.arange(half, dtype=np.int32), max_seq * num_chunks)

    # Raw offset chunk_index (D7)
    chunk_index = chunk.astype(np.int32) * cs

    global_dim = chunk.astype(np.int32) * cs + 2 * pair
    d_in_head = global_dim % head_dim
    pair_idx = d_in_head // 2
    inv_freq = (1.0 / (rope_theta ** (2.0 * pair_idx / head_dim))).astype(np.float64)
    angle = pos.astype(np.float64) * inv_freq
    cos_vals = np.cos(angle).astype(np.float32)
    sin_vals = np.sin(angle).astype(np.float32)

    flat_df = pd.DataFrame({
        "pos": pos,
        "chunk_index": chunk_index,
        "pair": pair,
        "cos_val": cos_vals,
        "sin_val": sin_vals,
    })
    con.register("_rope_flat", flat_df)
    con.execute("DROP TABLE IF EXISTS rope")
    con.execute(f"""
        CREATE TABLE rope AS
        SELECT
            pos AS row_index,
            chunk_index,
            array_agg(cos_val ORDER BY pair) AS cos,
            array_agg(sin_val ORDER BY pair) AS sin
        FROM _rope_flat
        GROUP BY pos, chunk_index
        ORDER BY row_index, chunk_index
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_rope_chunk ON rope(chunk_index)")
    cnt = con.execute("SELECT COUNT(*) FROM rope").fetchone()[0]
    print(f"  rope: {cnt} rows (computed from formula, theta={rope_theta})")


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_table_from_csv(
    con: duckdb.DuckDBPyConnection,
    csv_path: str,
    table_name: str,
    chunk_size: int,
) -> None:
    """Load a single CSV file into a DuckDB table."""
    schema = table_schema(table_name, chunk_size)
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(schema)

    # Column type specification for read_csv
    if _is_norm_table(table_name):
        # Decision D1: 1D norm has (chunk_index, v)
        col_types = "{'chunk_index': 'INTEGER', 'v': 'FLOAT[]'}"
    elif _is_moe_table(table_name):
        col_types = ("{'expert_id': 'INTEGER', 'row_index': 'INTEGER', "
                     "'chunk_index': 'INTEGER', 'v': 'FLOAT[]'}")
    else:
        col_types = "{'row_index': 'INTEGER', 'chunk_index': 'INTEGER', 'v': 'FLOAT[]'}"

    con.execute(
        f"INSERT INTO {table_name} "
        f"SELECT * FROM read_csv('{csv_path}', columns={col_types})"
    )

    count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"  {table_name}: {count} rows")

    # Index on chunk_index for MatMul join performance
    if not _is_norm_table(table_name):
        # Norm tables have chunk_index as PK already
        con.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_chunk "
            f"ON {table_name}(chunk_index)"
        )

    # MOE expert_id index (§3.2)
    if _is_moe_table(table_name):
        con.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_eid "
            f"ON {table_name}(expert_id)"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_all(
    csv_dir: str,
    db_path: str,
    config: ModelConfig,
) -> None:
    """Load all weight CSVs into a DuckDB database."""
    print(f"Opening DuckDB at {db_path}...")
    con = duckdb.connect(db_path)

    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    # Compute rope table from formula (bypasses CSV for precision)
    print(f"Computing rope table (theta={config.rope_theta})...")
    load_rope_from_formula(con, config)

    # Load all non-rope CSVs
    non_rope = [f for f in csv_files
                if os.path.splitext(os.path.basename(f))[0] != "rope"]
    print(f"Loading {len(non_rope)} weight tables from CSV...")
    for csv_path in non_rope:
        table_name = os.path.splitext(os.path.basename(csv_path))[0]
        load_table_from_csv(con, csv_path, table_name, config.chunk_size)

    # Embedding lookup index
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_embed_row "
        "ON embed_tokens(row_index)"
    )

    con.close()
    print(f"\nDone. Database written to {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load chunked CSV weights into DuckDB (§3.1.1)")
    parser.add_argument("--csv-dir", required=True)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--rope-theta", type=float, default=500000.0,
                        help="RoPE theta (500000 for Llama-3, 10000 for Llama-2)")
    args = parser.parse_args()

    config = ModelConfig.llama3_8b(chunk_size=args.chunk_size)
    config.rope_theta = args.rope_theta
    load_all(args.csv_dir, args.db_path, config)


if __name__ == "__main__":
    main()
