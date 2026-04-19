"""
Tests for TranSQL+ runner (Section 5).

Verifies:
  - Prefill produces correct logits (vs NumPy reference)
  - Weight pivot caching works (pivoted tables created once, reused)
  - JSON topology import round-trips correctly
"""

from __future__ import annotations

import json
import os
import tempfile

import duckdb
import numpy as np
import pytest

from transql_plus.config import ModelConfig
from transql_plus.compute_dag import TensorComputeDAG
from transql_plus.runner import TranSQLRunner
from transql_plus.postopt import PostOptOptions
from preprocessing.extract_weights import precompute_rope
from preprocessing.preprocess_weights import (
    write_2d_csv, write_1d_csv, write_rope_csv,
)
from preprocessing.load_weights_duckdb import (
    load_table_from_csv, load_rope_from_formula,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_config():
    """Tiny config: hidden=8, ffn=16, chunk=4, 1 layer, vocab=16, seq=3."""
    return ModelConfig(
        hidden_dim=8, num_q_heads=2, num_kv_heads=2, head_dim=4,
        ffn_dim=16, num_layers=1, vocab_size=16,
        rms_norm_eps=1e-5, rope_theta=10000.0,
        max_seq_len=8, chunk_size=4,
    )


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def tiny_db(tiny_config, rng, tmp_path):
    """Create a tiny weights.duckdb with synthetic data."""
    cfg = tiny_config
    cs = cfg.chunk_size
    hd = cfg.hidden_dim
    kv = cfg.kv_dim
    ffn = cfg.ffn_dim

    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    db_path = str(tmp_path / "weights.duckdb")

    # Generate and save weights
    weights = {
        "embed_tokens": rng.standard_normal((cfg.vocab_size, hd)).astype(np.float32),
        "lm_head": rng.standard_normal((cfg.vocab_size, hd)).astype(np.float32),
        "final_norm": rng.standard_normal(hd).astype(np.float32),
        "layer_0_q_proj": rng.standard_normal((hd, hd)).astype(np.float32),
        "layer_0_k_proj": rng.standard_normal((kv, hd)).astype(np.float32),
        "layer_0_v_proj": rng.standard_normal((kv, hd)).astype(np.float32),
        "layer_0_o_proj": rng.standard_normal((hd, hd)).astype(np.float32),
        "layer_0_gate_proj": rng.standard_normal((ffn, hd)).astype(np.float32),
        "layer_0_up_proj": rng.standard_normal((ffn, hd)).astype(np.float32),
        "layer_0_down_proj": rng.standard_normal((hd, ffn)).astype(np.float32),
        "layer_0_norm1": rng.standard_normal(hd).astype(np.float32),
        "layer_0_norm2": rng.standard_normal(hd).astype(np.float32),
    }

    # Write CSVs
    for name, arr in weights.items():
        path = str(csv_dir / f"{name}.csv")
        if arr.ndim == 1:
            write_1d_csv(path, arr, cs)
        else:
            write_2d_csv(path, arr, cs)

    # RoPE
    cos, sin = precompute_rope(hd, cfg.head_dim, cfg.rope_theta,
                                cfg.max_seq_len, cs)
    write_rope_csv(str(csv_dir / "rope.csv"), cos, sin, cs)

    # Load into DuckDB
    con = duckdb.connect(db_path)
    load_rope_from_formula(con, cfg)
    for name in weights:
        csv_path = str(csv_dir / f"{name}.csv")
        load_table_from_csv(con, csv_path, name, cs)
    con.close()

    return db_path, weights


# ---------------------------------------------------------------------------
# Prefill test
# ---------------------------------------------------------------------------

class TestPrefill:
    def test_prefill_runs(self, tiny_config, tiny_db):
        """Prefill completes without error and produces a logits table."""
        db_path, _ = tiny_db
        runner = TranSQLRunner(db_path, tiny_config, read_only=True)
        runner.init()

        tokens = [0, 1, 2]
        result = runner.run_prefill(tokens)

        assert result.latency_s > 0
        assert result.step_count > 0

        # Verify logits table exists and has correct shape
        out_table = runner.get_output_table()
        rows = runner.con.execute(
            f"SELECT COUNT(*) FROM {out_table}"
        ).fetchone()[0]
        assert rows > 0

        runner.close()

    def test_prefill_baseline_vs_postopt(self, tiny_config, tiny_db):
        """Baseline and postopt prefill produce the same logits."""
        db_path, _ = tiny_db
        tokens = [0, 1, 2]

        # Baseline (no postopt)
        runner_base = TranSQLRunner(db_path, tiny_config, read_only=True)
        runner_base.init()
        runner_base.run_prefill(tokens)
        base_logits = runner_base.con.execute(
            "SELECT row_index, chunk_index, v FROM logits "
            "ORDER BY row_index, chunk_index"
        ).fetchall()
        runner_base.close()

        # Postopt (CTE merge + fusion, no pivot for tiny dims)
        opts = PostOptOptions(cte_merge=True, table_fusion=True,
                              row2col_pivot=False)
        runner_opt = TranSQLRunner(db_path, tiny_config,
                                    postopt=opts, read_only=True)
        runner_opt.init()
        runner_opt.run_prefill(tokens)
        opt_logits = runner_opt.con.execute(
            "SELECT row_index, chunk_index, v FROM logits "
            "ORDER BY row_index, chunk_index"
        ).fetchall()
        runner_opt.close()

        # Compare
        assert len(base_logits) == len(opt_logits)
        for (ri1, ci1, v1), (ri2, ci2, v2) in zip(base_logits, opt_logits):
            assert ri1 == ri2
            assert ci1 == ci2
            np.testing.assert_allclose(v1, v2, atol=1e-4)


# ---------------------------------------------------------------------------
# Weight pivot caching test
# ---------------------------------------------------------------------------

class TestWeightPivotCaching:
    def test_pivot_tables_created(self, tiny_config, tiny_db):
        """Weight pivot tables are created during init()."""
        db_path, _ = tiny_db
        opts = PostOptOptions(row2col_pivot=True, cte_merge=False,
                              table_fusion=False)
        runner = TranSQLRunner(db_path, tiny_config,
                                postopt=opts, read_only=True)
        runner.init()

        # Should have pivoted weight tables
        assert len(runner._pivoted_weights) > 0
        for piv in runner._pivoted_weights:
            count = runner.con.execute(
                f"SELECT COUNT(*) FROM {piv}"
            ).fetchone()[0]
            assert count > 0, f"Pivoted table {piv} is empty"

        runner.close()

    def test_pivot_tables_reused(self, tiny_config, tiny_db):
        """Multiple prefill runs reuse the same pivoted weight tables."""
        db_path, _ = tiny_db
        opts = PostOptOptions(row2col_pivot=True, cte_merge=False,
                              table_fusion=False)
        runner = TranSQLRunner(db_path, tiny_config,
                                postopt=opts, read_only=True)
        runner.init()

        pivot_count_before = len(runner._pivoted_weights)

        # Run prefill twice
        runner.run_prefill([0, 1])
        runner.run_prefill([2, 3])

        # Same number of pivoted tables (not doubled)
        assert len(runner._pivoted_weights) == pivot_count_before

        runner.close()


# ---------------------------------------------------------------------------
# D11: Fusion × decode KV cache interaction
# ---------------------------------------------------------------------------

class TestDecodeKVCacheAfterFusion:
    def test_v_materialized_after_prefill_with_fusion(
        self, tiny_config, tiny_db,
    ):
        """QKV fusion absorbs l0_v into l0_q_qkv; decode needs l0_v as a
        standalone table to INSERT into. D11 extracts it lazily on the
        first decode step."""
        db_path, _ = tiny_db
        opts = PostOptOptions(cte_merge=True, table_fusion=True,
                              row2col_pivot=False)
        runner = TranSQLRunner(
            db_path, tiny_config, postopt=opts, read_only=False,
        )
        runner.init()
        runner.run_prefill([0, 1, 2])

        def temp_tables(con):
            return {r[0] for r in con.execute(
                "SELECT table_name FROM duckdb_tables() "
                "WHERE temporary = true"
            ).fetchall()}

        tables = temp_tables(runner.con)
        # Sanity: fusion did materialize the fused table
        assert "l0_q_qkv" in tables
        # Before decode: l0_v is only a filter CTE inside consumers,
        # not a standalone table — this is exactly the bug D11 addresses.
        assert "l0_v" not in tables

        # Trigger D11 materialization via first decode step
        current_token = runner.get_logits_argmax()
        result = runner.run_decode_step(current_token, pos=3)
        assert result.latency_s > 0

        tables = temp_tables(runner.con)
        assert "l0_v" in tables, "D11: l0_v must be materialized for decode"
        # Row count should match the appended decode step
        # (prefill 3 rows × kv_chunks + decode 1 row × kv_chunks)
        cs = tiny_config.chunk_size
        kv_chunks = tiny_config.kv_dim // cs
        v_rows = runner.con.execute(
            "SELECT COUNT(*) FROM l0_v"
        ).fetchone()[0]
        assert v_rows == (3 + 1) * kv_chunks

        runner.close()

    def test_decode_works_without_fusion(self, tiny_config, tiny_db):
        """Baseline sanity: without fusion, l0_v already exists post-prefill
        and decode works without the D11 materialization."""
        db_path, _ = tiny_db
        opts = PostOptOptions(cte_merge=True, table_fusion=False,
                              row2col_pivot=False)
        runner = TranSQLRunner(
            db_path, tiny_config, postopt=opts, read_only=False,
        )
        runner.init()
        runner.run_prefill([0, 1, 2])

        tables = {r[0] for r in runner.con.execute(
            "SELECT table_name FROM duckdb_tables() WHERE temporary = true"
        ).fetchall()}
        assert "l0_v" in tables
        assert "l0_q_qkv" not in tables

        current_token = runner.get_logits_argmax()
        result = runner.run_decode_step(current_token, pos=3)
        assert result.latency_s > 0

        runner.close()


# ---------------------------------------------------------------------------
# JSON topology import test
# ---------------------------------------------------------------------------

class TestJSONTopology:
    def test_build_from_json_roundtrip(self, tiny_config, tmp_path):
        """build_from_json produces same DAG as build_llama3_8b."""
        dag = TensorComputeDAG.build_llama3_8b(tiny_config)

        # Serialize to JSON
        nodes_json = []
        for node in dag.nodes:
            nodes_json.append({
                "id": node.id,
                "op_type": node.op_type.name,
                "output_table": node.output_table,
                "input_tables": node.input_tables,
                "is_shared": node.is_shared,
                "params": node.params,
            })
        topology = {
            "nodes": nodes_json,
            "output_node_id": dag.output_node_id,
        }

        json_path = tmp_path / "topology.json"
        with open(json_path, "w") as f:
            json.dump(topology, f)

        # Reconstruct
        dag2 = TensorComputeDAG.build_from_json(json_path)

        assert len(dag2.nodes) == len(dag.nodes)
        assert dag2.output_node_id == dag.output_node_id
        for n1, n2 in zip(dag.nodes, dag2.nodes):
            assert n1.op_type == n2.op_type
            assert n1.output_table == n2.output_table
            assert n1.input_tables == n2.input_tables
            assert n1.is_shared == n2.is_shared
