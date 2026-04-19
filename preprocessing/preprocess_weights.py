"""
Convert .npy weight files to chunked CSV format.

Paper reference: Section 3.1.1, Algorithm 1.

Algorithm 1 chunk-based representation:
    for i in range(M):
        for c in range(0, N, chunk_size):
            chunk = matrix[i, c : c + chunk_size]
            table.append((row_index=i, chunk_index=c, vector=chunk))

Conventions:
    - chunk_index uses raw offset (0, 32, 64, ...) per Algorithm 1 (Decision D7)
    - 1D norm weights use (chunk_index, v) with no row_index (Decision D1)
    - Dimensions from ModelConfig, not hardcoded (Decision D6)

CSV schemas:
    2D weight: row_index, chunk_index, v
    1D norm:   chunk_index, v
    RoPE:      row_index, chunk_index, cos, sin

Usage:
    python -m preprocessing.preprocess_weights \\
        --npy-dir weights_npy --csv-dir weights_csv \\
        --config llama3-8b [--chunk-size 32]
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Iterator

import numpy as np

from transql_plus.config import ModelConfig


def format_list(arr: np.ndarray) -> str:
    """Format a 1D numpy array as a DuckDB list literal: [1.0, 2.0, ...]"""
    return "[" + ", ".join(f"{x:.8g}" for x in arr) + "]"


# ---------------------------------------------------------------------------
# Chunking generators (Algorithm 1)
# ---------------------------------------------------------------------------

def chunk_2d(
    weight: np.ndarray, chunk_size: int,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield (row_index, chunk_index, chunk) for a 2D weight [out_dim, in_dim].

    Implements Algorithm 1 from §3.1.1.  The input weight is assumed to be
    already in [out_dim, contracted_dim] layout — i.e. the preprocessing
    transpose (§3.1: "storing column vectors as row vectors") has already
    been applied in extract_weights.py.  Chunking along axis 1 therefore
    produces chunks along the contracted dimension, so at runtime the SQL
    JOIN on chunk_index aligns matching chunks for direct dot products.

    chunk_index is the raw byte offset along in_dim (Decision D7).
    """
    out_dim, in_dim = weight.shape
    assert in_dim % chunk_size == 0, \
        f"in_dim {in_dim} not divisible by chunk_size {chunk_size}"
    for row in range(out_dim):
        for c in range(0, in_dim, chunk_size):
            yield row, c, weight[row, c:c + chunk_size]


def chunk_1d(
    weight: np.ndarray, chunk_size: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (chunk_index, chunk) for a 1D weight [dim].

    No row_index for 1D norms (Decision D1).
    chunk_index is the raw byte offset (Decision D7).
    """
    dim = weight.shape[0]
    assert dim % chunk_size == 0, \
        f"dim {dim} not divisible by chunk_size {chunk_size}"
    for c in range(0, dim, chunk_size):
        yield c, weight[c:c + chunk_size]


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_2d_csv(path: str, weight: np.ndarray, chunk_size: int) -> None:
    """Write a 2D weight to CSV: row_index, chunk_index, v."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["row_index", "chunk_index", "v"])
        for row_index, chunk_index, chunk in chunk_2d(weight, chunk_size):
            writer.writerow([row_index, chunk_index, format_list(chunk)])


def write_1d_csv(path: str, weight: np.ndarray, chunk_size: int) -> None:
    """Write a 1D norm weight to CSV: chunk_index, v (Decision D1)."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["chunk_index", "v"])
        for chunk_index, chunk in chunk_1d(weight, chunk_size):
            writer.writerow([chunk_index, format_list(chunk)])


def write_rope_csv(
    path: str,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    chunk_size: int,
) -> None:
    """Write RoPE cos/sin tables to CSV: row_index, chunk_index, cos, sin.

    Input shapes: [max_seq_len, num_chunks, chunk_size//2].
    chunk_index is raw offset (Decision D7).
    """
    max_seq, num_chunks, half = cos_table.shape
    assert half == chunk_size // 2
    assert cos_table.shape == sin_table.shape

    with open(path, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["row_index", "chunk_index", "cos", "sin"])
        for pos in range(max_seq):
            for c in range(num_chunks):
                chunk_index = c * chunk_size  # raw offset (D7)
                writer.writerow([
                    pos, chunk_index,
                    format_list(cos_table[pos, c]),
                    format_list(sin_table[pos, c]),
                ])


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def preprocess_all(
    npy_dir: str,
    csv_dir: str,
    config: ModelConfig,
) -> None:
    """Convert all .npy weight files to chunked CSV format."""
    os.makedirs(csv_dir, exist_ok=True)
    cs = config.chunk_size

    # 1D norm weights: (chunk_index, v) — Decision D1
    norm_names = (
        ["final_norm"]
        + [f"layer_{l}_norm1" for l in range(config.num_layers)]
        + [f"layer_{l}_norm2" for l in range(config.num_layers)]
    )
    for name in norm_names:
        npy_path = os.path.join(npy_dir, name + ".npy")
        if not os.path.exists(npy_path):
            print(f"  SKIP (not found): {name}")
            continue
        arr = np.load(npy_path).astype(np.float32)
        assert arr.ndim == 1, f"{name}: expected 1D, got shape {arr.shape}"
        write_1d_csv(os.path.join(csv_dir, name + ".csv"), arr, cs)
        print(f"  {name}: {arr.shape} → {arr.shape[0] // cs} chunks")

    # 2D weights: (row_index, chunk_index, v)
    two_d_names = (
        ["embed_tokens", "lm_head"]
        + [f"layer_{l}_{w}"
           for l in range(config.num_layers)
           for w in ("q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj")]
    )
    for name in two_d_names:
        npy_path = os.path.join(npy_dir, name + ".npy")
        if not os.path.exists(npy_path):
            print(f"  SKIP (not found): {name}")
            continue
        arr = np.load(npy_path).astype(np.float32)
        assert arr.ndim == 2, f"{name}: expected 2D, got shape {arr.shape}"
        write_2d_csv(os.path.join(csv_dir, name + ".csv"), arr, cs)
        print(f"  {name}: {arr.shape} → {arr.shape[0]} rows x "
              f"{arr.shape[1] // cs} chunks/row")

    # RoPE tables
    cos_path = os.path.join(npy_dir, "rope_cos.npy")
    sin_path = os.path.join(npy_dir, "rope_sin.npy")
    if os.path.exists(cos_path) and os.path.exists(sin_path):
        cos_table = np.load(cos_path).astype(np.float32)
        sin_table = np.load(sin_path).astype(np.float32)
        write_rope_csv(os.path.join(csv_dir, "rope.csv"),
                       cos_table, sin_table, cs)
        print(f"  rope: {cos_table.shape[0]} positions x "
              f"{cos_table.shape[1]} chunks → rope.csv")
    else:
        print("  SKIP rope: rope_cos.npy or rope_sin.npy not found")

    print(f"\nDone. CSVs written to {csv_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert .npy weights to chunked CSV (Algorithm 1)")
    parser.add_argument("--npy-dir", required=True,
                        help="Directory containing .npy weight files")
    parser.add_argument("--csv-dir", required=True,
                        help="Output directory for CSV files")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=32,
                        help="Number of transformer layers (default: 32)")
    args = parser.parse_args()

    config = ModelConfig.llama3_8b(chunk_size=args.chunk_size)
    # Allow overriding num_layers for partial extractions
    config.num_layers = args.num_layers
    preprocess_all(args.npy_dir, args.csv_dir, config)


if __name__ == "__main__":
    main()
