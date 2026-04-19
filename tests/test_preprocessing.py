"""
Tests for Phase 2: Preprocessing (§3.1).

Verifies:
  - extract_weights: RoPE precomputation matches formula
  - preprocess_weights: Algorithm 1 chunking with raw offset (D7), 1D norms (D1)
  - load_weights_duckdb: CSV → DuckDB round-trip, schema correctness
"""

from __future__ import annotations

import csv
import os
import tempfile

import duckdb
import numpy as np
import pytest

from transql_plus.config import ModelConfig
from preprocessing.extract_weights import precompute_rope
from preprocessing.preprocess_weights import (
    chunk_2d, chunk_1d, format_list,
    write_2d_csv, write_1d_csv, write_rope_csv,
    preprocess_all,
)
from preprocessing.load_weights_duckdb import (
    table_schema, load_table_from_csv, load_rope_from_formula, load_all,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_config():
    """Small config for fast tests: hidden=8, ffn=16, chunk=4, 1 layer."""
    return ModelConfig(
        hidden_dim=8, num_q_heads=2, num_kv_heads=2, head_dim=4,
        ffn_dim=16, num_layers=1, vocab_size=16,
        rms_norm_eps=1e-5, rope_theta=10000.0,
        max_seq_len=4, chunk_size=4,
    )


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# extract_weights: RoPE precomputation
# ---------------------------------------------------------------------------

class TestRoPEPrecomputation:
    def test_shape(self, tiny_config):
        cos, sin = precompute_rope(
            tiny_config.hidden_dim, tiny_config.head_dim,
            tiny_config.rope_theta, tiny_config.max_seq_len,
            tiny_config.chunk_size,
        )
        num_chunks = tiny_config.hidden_dim // tiny_config.chunk_size
        half = tiny_config.chunk_size // 2
        assert cos.shape == (tiny_config.max_seq_len, num_chunks, half)
        assert sin.shape == cos.shape

    def test_position_zero(self, tiny_config):
        """At position 0, all angles are 0 → cos=1, sin=0."""
        cos, sin = precompute_rope(
            tiny_config.hidden_dim, tiny_config.head_dim,
            tiny_config.rope_theta, tiny_config.max_seq_len,
            tiny_config.chunk_size,
        )
        np.testing.assert_allclose(cos[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(sin[0], 0.0, atol=1e-6)

    def test_matches_formula(self, tiny_config):
        """Spot-check a specific (pos, chunk, pair) against manual calculation."""
        cos, sin = precompute_rope(
            tiny_config.hidden_dim, tiny_config.head_dim,
            tiny_config.rope_theta, tiny_config.max_seq_len,
            tiny_config.chunk_size,
        )
        cs = tiny_config.chunk_size
        hd = tiny_config.head_dim
        theta = tiny_config.rope_theta

        pos, c, i = 2, 1, 0  # position=2, chunk=1, pair=0
        global_dim = c * cs + 2 * i
        d = global_dim % hd
        pair_idx = d // 2
        angle = pos / (theta ** (2.0 * pair_idx / hd))
        expected_cos = np.cos(angle)
        expected_sin = np.sin(angle)

        np.testing.assert_allclose(cos[pos, c, i], expected_cos, atol=1e-5)
        np.testing.assert_allclose(sin[pos, c, i], expected_sin, atol=1e-5)


# ---------------------------------------------------------------------------
# preprocess_weights: chunking
# ---------------------------------------------------------------------------

class TestChunking:
    def test_chunk_2d_raw_offset(self, rng):
        """chunk_index is raw offset (0, 4, 8, ...) per Decision D7."""
        arr = rng.standard_normal((3, 8)).astype(np.float32)
        chunks = list(chunk_2d(arr, chunk_size=4))
        # 3 rows x 2 chunks = 6 entries
        assert len(chunks) == 6
        # First row
        assert chunks[0][0] == 0 and chunks[0][1] == 0  # row=0, offset=0
        assert chunks[1][0] == 0 and chunks[1][1] == 4  # row=0, offset=4
        # Second row
        assert chunks[2][0] == 1 and chunks[2][1] == 0
        np.testing.assert_array_equal(chunks[0][2], arr[0, 0:4])
        np.testing.assert_array_equal(chunks[1][2], arr[0, 4:8])

    def test_chunk_1d_no_row_index(self, rng):
        """1D norm chunking has no row_index (Decision D1)."""
        arr = rng.standard_normal(8).astype(np.float32)
        chunks = list(chunk_1d(arr, chunk_size=4))
        assert len(chunks) == 2
        # Each entry is (chunk_index, chunk_data) — no row_index
        assert chunks[0][0] == 0  # raw offset
        assert chunks[1][0] == 4
        np.testing.assert_array_equal(chunks[0][1], arr[0:4])
        np.testing.assert_array_equal(chunks[1][1], arr[4:8])

    def test_format_list(self):
        arr = np.array([1.0, 2.5, -3.0], dtype=np.float32)
        result = format_list(arr)
        assert result.startswith("[")
        assert result.endswith("]")
        assert "1" in result and "2.5" in result and "-3" in result


class TestCSVWriters:
    def test_write_2d_csv(self, rng, tmp_dir):
        arr = rng.standard_normal((2, 8)).astype(np.float32)
        path = os.path.join(tmp_dir, "test_2d.csv")
        write_2d_csv(path, arr, chunk_size=4)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 4  # 2 rows x 2 chunks
        assert set(rows[0].keys()) == {"row_index", "chunk_index", "v"}
        # Raw offset check
        assert rows[0]["chunk_index"] == "0"
        assert rows[1]["chunk_index"] == "4"

    def test_write_1d_csv(self, rng, tmp_dir):
        arr = rng.standard_normal(8).astype(np.float32)
        path = os.path.join(tmp_dir, "test_1d.csv")
        write_1d_csv(path, arr, chunk_size=4)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        # Decision D1: no row_index column
        assert set(rows[0].keys()) == {"chunk_index", "v"}
        assert rows[0]["chunk_index"] == "0"
        assert rows[1]["chunk_index"] == "4"

    def test_write_rope_csv(self, tiny_config, tmp_dir):
        cos, sin = precompute_rope(
            tiny_config.hidden_dim, tiny_config.head_dim,
            tiny_config.rope_theta, tiny_config.max_seq_len,
            tiny_config.chunk_size,
        )
        path = os.path.join(tmp_dir, "rope.csv")
        write_rope_csv(path, cos, sin, tiny_config.chunk_size)

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        num_chunks = tiny_config.hidden_dim // tiny_config.chunk_size
        assert len(rows) == tiny_config.max_seq_len * num_chunks
        assert set(rows[0].keys()) == {"row_index", "chunk_index", "cos", "sin"}
        # Raw offset for rope chunk_index
        assert rows[0]["chunk_index"] == "0"
        if num_chunks > 1:
            assert rows[1]["chunk_index"] == str(tiny_config.chunk_size)


# ---------------------------------------------------------------------------
# load_weights_duckdb: schema and round-trip
# ---------------------------------------------------------------------------

class TestTableSchema:
    def test_2d_schema(self):
        ddl = table_schema("layer_0_q_proj", 32)
        assert "row_index" in ddl
        assert "chunk_index" in ddl
        assert "FLOAT[32]" in ddl
        assert "PRIMARY KEY" in ddl

    def test_1d_norm_schema(self):
        """Decision D1: norm tables have no row_index."""
        ddl = table_schema("layer_0_norm1", 32)
        assert "row_index" not in ddl
        assert "chunk_index" in ddl
        assert "PRIMARY KEY" in ddl

    def test_rope_schema(self):
        ddl = table_schema("rope", 32)
        assert "cos FLOAT[16]" in ddl
        assert "sin FLOAT[16]" in ddl


class TestDuckDBRoundTrip:
    """Full round-trip: numpy → CSV → DuckDB → read back → compare."""

    def test_2d_roundtrip(self, rng, tmp_dir):
        arr = rng.standard_normal((3, 8)).astype(np.float32)
        csv_path = os.path.join(tmp_dir, "test_weight.csv")
        write_2d_csv(csv_path, arr, chunk_size=4)

        con = duckdb.connect(":memory:")
        load_table_from_csv(con, csv_path, "test_weight", chunk_size=4)

        # Read back and reconstruct
        rows = con.execute(
            "SELECT row_index, chunk_index, v FROM test_weight "
            "ORDER BY row_index, chunk_index"
        ).fetchall()
        assert len(rows) == 6  # 3 rows x 2 chunks

        reconstructed = np.zeros_like(arr)
        for ri, ci, v in rows:
            reconstructed[ri, ci:ci + 4] = v
        np.testing.assert_allclose(reconstructed, arr, atol=1e-6)
        con.close()

    def test_1d_roundtrip(self, rng, tmp_dir):
        arr = rng.standard_normal(8).astype(np.float32)
        csv_path = os.path.join(tmp_dir, "final_norm.csv")
        write_1d_csv(csv_path, arr, chunk_size=4)

        con = duckdb.connect(":memory:")
        load_table_from_csv(con, csv_path, "final_norm", chunk_size=4)

        rows = con.execute(
            "SELECT chunk_index, v FROM final_norm ORDER BY chunk_index"
        ).fetchall()
        assert len(rows) == 2

        reconstructed = np.zeros_like(arr)
        for ci, v in rows:
            reconstructed[ci:ci + 4] = v
        np.testing.assert_allclose(reconstructed, arr, atol=1e-6)
        con.close()

    def test_rope_from_formula(self, tiny_config):
        """Rope loaded from formula matches precompute_rope output."""
        con = duckdb.connect(":memory:")
        load_rope_from_formula(con, tiny_config)

        cos_ref, sin_ref = precompute_rope(
            tiny_config.hidden_dim, tiny_config.head_dim,
            tiny_config.rope_theta, tiny_config.max_seq_len,
            tiny_config.chunk_size,
        )

        rows = con.execute(
            "SELECT row_index, chunk_index, cos, sin FROM rope "
            "ORDER BY row_index, chunk_index"
        ).fetchall()

        cs = tiny_config.chunk_size
        num_chunks = tiny_config.hidden_dim // cs
        assert len(rows) == tiny_config.max_seq_len * num_chunks

        for pos_idx, (ri, ci, cos_v, sin_v) in enumerate(rows):
            pos = ri
            c = ci // cs  # convert raw offset back to chunk index
            np.testing.assert_allclose(
                cos_v, cos_ref[pos, c], atol=1e-5,
                err_msg=f"cos mismatch at pos={pos}, chunk={c}",
            )
            np.testing.assert_allclose(
                sin_v, sin_ref[pos, c], atol=1e-5,
                err_msg=f"sin mismatch at pos={pos}, chunk={c}",
            )
        con.close()


class TestFullPipeline:
    """End-to-end: generate synthetic .npy → preprocess → load → verify."""

    def test_full_pipeline(self, tiny_config, rng, tmp_dir):
        npy_dir = os.path.join(tmp_dir, "npy")
        csv_dir = os.path.join(tmp_dir, "csv")
        db_path = os.path.join(tmp_dir, "test.duckdb")
        os.makedirs(npy_dir)

        cs = tiny_config.chunk_size
        hd = tiny_config.hidden_dim
        kv = tiny_config.kv_dim
        ffn = tiny_config.ffn_dim

        # Generate synthetic weights
        weights = {
            "embed_tokens": rng.standard_normal((tiny_config.vocab_size, hd)),
            "lm_head": rng.standard_normal((tiny_config.vocab_size, hd)),
            "final_norm": rng.standard_normal(hd),
            "layer_0_q_proj": rng.standard_normal((hd, hd)),
            "layer_0_k_proj": rng.standard_normal((kv, hd)),
            "layer_0_v_proj": rng.standard_normal((kv, hd)),
            "layer_0_o_proj": rng.standard_normal((hd, hd)),
            "layer_0_gate_proj": rng.standard_normal((ffn, hd)),
            "layer_0_up_proj": rng.standard_normal((ffn, hd)),
            "layer_0_down_proj": rng.standard_normal((hd, ffn)),
            "layer_0_norm1": rng.standard_normal(hd),
            "layer_0_norm2": rng.standard_normal(hd),
        }
        for name, arr in weights.items():
            np.save(os.path.join(npy_dir, name + ".npy"),
                    arr.astype(np.float32))

        # RoPE tables
        cos, sin = precompute_rope(hd, tiny_config.head_dim,
                                   tiny_config.rope_theta,
                                   tiny_config.max_seq_len, cs)
        np.save(os.path.join(npy_dir, "rope_cos.npy"), cos)
        np.save(os.path.join(npy_dir, "rope_sin.npy"), sin)

        # Preprocess: npy → csv
        preprocess_all(npy_dir, csv_dir, tiny_config)

        # Verify CSVs exist
        assert os.path.exists(os.path.join(csv_dir, "embed_tokens.csv"))
        assert os.path.exists(os.path.join(csv_dir, "final_norm.csv"))
        assert os.path.exists(os.path.join(csv_dir, "layer_0_q_proj.csv"))
        assert os.path.exists(os.path.join(csv_dir, "rope.csv"))

        # Load: csv → duckdb
        load_all(csv_dir, db_path, tiny_config)

        # Verify tables exist and have correct data
        con = duckdb.connect(db_path, read_only=True)

        # Check 2D weight round-trip
        q_rows = con.execute(
            "SELECT row_index, chunk_index, v FROM layer_0_q_proj "
            "ORDER BY row_index, chunk_index"
        ).fetchall()
        q_orig = weights["layer_0_q_proj"].astype(np.float32)
        q_recon = np.zeros_like(q_orig)
        for ri, ci, v in q_rows:
            q_recon[ri, ci:ci + cs] = v
        np.testing.assert_allclose(q_recon, q_orig, atol=1e-5)

        # Check 1D norm round-trip (Decision D1: no row_index)
        norm_rows = con.execute(
            "SELECT chunk_index, v FROM final_norm ORDER BY chunk_index"
        ).fetchall()
        norm_orig = weights["final_norm"].astype(np.float32)
        norm_recon = np.zeros_like(norm_orig)
        for ci, v in norm_rows:
            norm_recon[ci:ci + cs] = v
        np.testing.assert_allclose(norm_recon, norm_orig, atol=1e-5)

        # Check rope table exists
        rope_count = con.execute("SELECT COUNT(*) FROM rope").fetchone()[0]
        expected_rope = tiny_config.max_seq_len * (hd // cs)
        assert rope_count == expected_rope

        con.close()
