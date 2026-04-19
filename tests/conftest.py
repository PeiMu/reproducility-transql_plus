"""
Shared test fixtures for TranSQL+ unit tests.

Provides helpers for creating in-memory DuckDB connections, loading
synthetic weight tables, and running SQL step sequences.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pytest


@pytest.fixture
def conn():
    """Fresh in-memory DuckDB connection, closed after test."""
    con = duckdb.connect(":memory:")
    yield con
    con.close()


# ---------------------------------------------------------------------------
# Helpers for loading synthetic data (raw-offset chunk_index, Decision D7)
# ---------------------------------------------------------------------------

def load_2d(con: duckdb.DuckDBPyConnection,
            table_name: str, arr: np.ndarray,
            chunk_size: int) -> None:
    """Load a 2D numpy array into a chunked DuckDB table.

    Schema: (row_index INTEGER, chunk_index INTEGER, v FLOAT[chunk_size])
    chunk_index uses raw offset (0, 32, 64, ...) per Algorithm 1 / Decision D7.
    """
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
            offset = c * chunk_size    # raw offset (Decision D7)
            chunk = arr[r, c * chunk_size:(c + 1) * chunk_size].tolist()
            rows.append((r, offset, chunk))

    con.executemany(
        f"INSERT INTO {table_name} VALUES (?, ?, ?::FLOAT[{chunk_size}])",
        rows,
    )


def load_norm_weight(con: duckdb.DuckDBPyConnection,
                     table_name: str, arr: np.ndarray,
                     chunk_size: int) -> None:
    """Load a 1D norm weight into DuckDB.

    Schema: (chunk_index INTEGER PRIMARY KEY, v FLOAT[chunk_size])
    No row_index — Decision D1.
    """
    assert arr.ndim == 1
    dim = arr.shape[0]
    assert dim % chunk_size == 0
    n_chunks = dim // chunk_size

    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(
        f"CREATE TABLE {table_name} "
        f"(chunk_index INTEGER PRIMARY KEY, v FLOAT[{chunk_size}])"
    )

    rows = []
    for c in range(n_chunks):
        offset = c * chunk_size
        chunk = arr[c * chunk_size:(c + 1) * chunk_size].tolist()
        rows.append((offset, chunk))

    con.executemany(
        f"INSERT INTO {table_name} VALUES (?, ?::FLOAT[{chunk_size}])",
        rows,
    )


def load_rope_table(con: duckdb.DuckDBPyConnection,
                    cos_arr: np.ndarray, sin_arr: np.ndarray,
                    chunk_size: int, table_name: str = "rope") -> None:
    """Load precomputed RoPE cos/sin tables.

    cos_arr, sin_arr: shape [max_seq_len, num_chunks, half_chunk]
    Schema: (row_index INTEGER, chunk_index INTEGER,
             cos FLOAT[half], sin FLOAT[half])
    """
    half = chunk_size // 2
    max_seq, n_chunks, h = cos_arr.shape
    assert h == half

    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(
        f"CREATE TABLE {table_name} "
        f"(row_index INTEGER, chunk_index INTEGER, "
        f"cos FLOAT[{half}], sin FLOAT[{half}], "
        f"PRIMARY KEY (row_index, chunk_index))"
    )

    rows = []
    for pos in range(max_seq):
        for c in range(n_chunks):
            offset = c * chunk_size
            cos_chunk = cos_arr[pos, c].tolist()
            sin_chunk = sin_arr[pos, c].tolist()
            rows.append((pos, offset, cos_chunk, sin_chunk))

    con.executemany(
        f"INSERT INTO {table_name} VALUES (?, ?, ?::FLOAT[{half}], ?::FLOAT[{half}])",
        rows,
    )


def read_2d(con: duckdb.DuckDBPyConnection,
            table_name: str, out_dim: int, in_dim: int,
            chunk_size: int) -> np.ndarray:
    """Read a chunked table back into a 2D numpy array."""
    result = con.execute(
        f"SELECT row_index, chunk_index, v FROM {table_name} "
        f"ORDER BY row_index, chunk_index"
    ).fetchall()

    arr = np.zeros((out_dim, in_dim), dtype=np.float32)
    for row_idx, chunk_idx, v in result:
        col_start = chunk_idx  # raw offset (Decision D7)
        arr[row_idx, col_start:col_start + chunk_size] = v
    return arr


def read_scalar_table(con: duckdb.DuckDBPyConnection,
                      table_name: str) -> list[tuple]:
    """Read a scalar-layout table (q_tok, k_tok, head_id, value)."""
    return con.execute(
        f"SELECT * FROM {table_name} ORDER BY 1, 2, 3"
    ).fetchall()


def run_steps(con: duckdb.DuckDBPyConnection,
              steps: list[tuple[str, str]]) -> None:
    """Execute a list of (sql_body, table_name) steps as CREATE TEMP TABLE."""
    for sql, name in steps:
        con.execute(f"CREATE TEMP TABLE {name} AS ({sql})")
