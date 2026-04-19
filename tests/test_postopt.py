"""
Tests for Section 4 post-optimisations.

Verifies that each optimisation produces numerically identical results
to the baseline dag_to_sql() expansion.

Also verifies the paper's claim: CTE merging reduces per-layer
materialised intermediates from 38 to 7 (§4.1).

Source: AQP_middleware/transql/python/unit_tests/test_postopt.py
"""

import numpy as np
import pytest

from tests.conftest import (
    load_2d, load_norm_weight, load_rope_table, read_2d, run_steps,
)
from transql_plus.config import ModelConfig
from transql_plus.compute_dag import TensorComputeDAG
from transql_plus.dag_to_sql import dag_to_sql
from transql_plus.postopt import (
    PostOptOptions, postopt_dag_to_sql,
    fused_qkv_sql, fused_gate_up_sql, pivoted_matmul_sql,
)
from transql_plus.sql_templates import matmul_sql

# ---------------------------------------------------------------------------
# Test parameters (same as test_sql_templates.py)
# ---------------------------------------------------------------------------
CHUNK_SIZE   = 2
SEQ_LEN      = 3
HIDDEN_DIM   = 4
KV_DIM       = 2
FFN_DIM      = 8
NUM_Q_HEADS  = 2
NUM_KV_HEADS = 1
HEAD_DIM     = 2
EPS          = 1e-5
ATOL         = 1e-3
RTOL         = 1e-3

rng = np.random.default_rng(999)


def make_config() -> ModelConfig:
    return ModelConfig(
        hidden_dim=HIDDEN_DIM,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        ffn_dim=FFN_DIM,
        num_layers=1,
        vocab_size=16,
        rms_norm_eps=EPS,
        rope_theta=10000.0,
        max_seq_len=SEQ_LEN,
        chunk_size=CHUNK_SIZE,
    )


def load_layer_weights(conn, rng_inst):
    """Load synthetic layer-0 weights and return numpy arrays."""
    weights = {
        "norm1":     rng_inst.standard_normal(HIDDEN_DIM).astype(np.float32) + 1.0,
        "q_proj":    rng_inst.standard_normal((HIDDEN_DIM, HIDDEN_DIM)).astype(np.float32) * 0.1,
        "k_proj":    rng_inst.standard_normal((KV_DIM, HIDDEN_DIM)).astype(np.float32) * 0.1,
        "v_proj":    rng_inst.standard_normal((KV_DIM, HIDDEN_DIM)).astype(np.float32) * 0.1,
        "o_proj":    rng_inst.standard_normal((HIDDEN_DIM, HIDDEN_DIM)).astype(np.float32) * 0.1,
        "norm2":     rng_inst.standard_normal(HIDDEN_DIM).astype(np.float32) + 1.0,
        "gate_proj": rng_inst.standard_normal((FFN_DIM, HIDDEN_DIM)).astype(np.float32) * 0.1,
        "up_proj":   rng_inst.standard_normal((FFN_DIM, HIDDEN_DIM)).astype(np.float32) * 0.1,
        "down_proj": rng_inst.standard_normal((HIDDEN_DIM, FFN_DIM)).astype(np.float32) * 0.1,
    }

    for name in ("q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"):
        load_2d(conn, f"layer_0_{name}", weights[name], CHUNK_SIZE)

    load_norm_weight(conn, "layer_0_norm1", weights["norm1"], CHUNK_SIZE)
    load_norm_weight(conn, "layer_0_norm2", weights["norm2"], CHUNK_SIZE)

    half = CHUNK_SIZE // 2
    n_chunks = HIDDEN_DIM // CHUNK_SIZE
    cos_arr = rng_inst.uniform(0.5, 1.0, (SEQ_LEN, n_chunks, half)).astype(np.float32)
    sin_arr = rng_inst.uniform(0.0, 0.5, (SEQ_LEN, n_chunks, half)).astype(np.float32)
    load_rope_table(conn, cos_arr, sin_arr, CHUNK_SIZE, "rope")

    return weights


# ===========================================================================
# §4.2  Table Fusion tests
# ===========================================================================

class TestFusedQKV:
    """Fused QKV produces single table; filtering by flag matches separate MatMuls."""

    def test_matches_separate(self, conn):
        act = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
        q_w = rng.standard_normal((HIDDEN_DIM, HIDDEN_DIM)).astype(np.float32) * 0.1
        k_w = rng.standard_normal((KV_DIM, HIDDEN_DIM)).astype(np.float32) * 0.1
        v_w = rng.standard_normal((KV_DIM, HIDDEN_DIM)).astype(np.float32) * 0.1

        load_2d(conn, "act", act, CHUNK_SIZE)
        load_2d(conn, "q_w", q_w, CHUNK_SIZE)
        load_2d(conn, "k_w", k_w, CHUNK_SIZE)
        load_2d(conn, "v_w", v_w, CHUNK_SIZE)

        # Baseline: separate matmuls
        run_steps(conn, matmul_sql("act", "q_w", "q_sep", CHUNK_SIZE))
        run_steps(conn, matmul_sql("act", "k_w", "k_sep", CHUNK_SIZE))
        run_steps(conn, matmul_sql("act", "v_w", "v_sep", CHUNK_SIZE))

        q_sep = read_2d(conn, "q_sep", SEQ_LEN, HIDDEN_DIM, CHUNK_SIZE)
        k_sep = read_2d(conn, "k_sep", SEQ_LEN, KV_DIM, CHUNK_SIZE)
        v_sep = read_2d(conn, "v_sep", SEQ_LEN, KV_DIM, CHUNK_SIZE)

        # Fused: single-table output with flag column
        fused_steps = fused_qkv_sql(
            "act", "q_w", "k_w", "v_w",
            "qkv_fused",
            HIDDEN_DIM, KV_DIM, CHUNK_SIZE,
        )
        run_steps(conn, fused_steps)

        # Extract Q, K, V by filtering on flag
        conn.execute(
            "CREATE TEMP TABLE q_fus AS "
            "SELECT row_index, chunk_index, v "
            "FROM qkv_fused WHERE flag = 'Q'"
        )
        conn.execute(
            "CREATE TEMP TABLE k_fus AS "
            "SELECT row_index, chunk_index, v "
            "FROM qkv_fused WHERE flag = 'K'"
        )
        conn.execute(
            "CREATE TEMP TABLE v_fus AS "
            "SELECT row_index, chunk_index, v "
            "FROM qkv_fused WHERE flag = 'V'"
        )

        q_fus = read_2d(conn, "q_fus", SEQ_LEN, HIDDEN_DIM, CHUNK_SIZE)
        k_fus = read_2d(conn, "k_fus", SEQ_LEN, KV_DIM, CHUNK_SIZE)
        v_fus = read_2d(conn, "v_fus", SEQ_LEN, KV_DIM, CHUNK_SIZE)

        np.testing.assert_allclose(q_fus, q_sep, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(k_fus, k_sep, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(v_fus, v_sep, atol=ATOL, rtol=RTOL)


class TestFusedGateUp:
    """Fused gate+up produces single table; filtering by flag matches separate MatMuls."""

    def test_matches_separate(self, conn):
        act = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
        g_w = rng.standard_normal((FFN_DIM, HIDDEN_DIM)).astype(np.float32) * 0.1
        u_w = rng.standard_normal((FFN_DIM, HIDDEN_DIM)).astype(np.float32) * 0.1

        load_2d(conn, "act", act, CHUNK_SIZE)
        load_2d(conn, "g_w", g_w, CHUNK_SIZE)
        load_2d(conn, "u_w", u_w, CHUNK_SIZE)

        # Baseline
        run_steps(conn, matmul_sql("act", "g_w", "g_sep", CHUNK_SIZE))
        run_steps(conn, matmul_sql("act", "u_w", "u_sep", CHUNK_SIZE))
        g_sep = read_2d(conn, "g_sep", SEQ_LEN, FFN_DIM, CHUNK_SIZE)
        u_sep = read_2d(conn, "u_sep", SEQ_LEN, FFN_DIM, CHUNK_SIZE)

        # Fused: single-table output with flag column
        fused_steps = fused_gate_up_sql(
            "act", "g_w", "u_w",
            "gateup_fused",
            FFN_DIM, CHUNK_SIZE,
        )
        run_steps(conn, fused_steps)

        # Extract gate, up by filtering on flag
        conn.execute(
            "CREATE TEMP TABLE g_fus AS "
            "SELECT row_index, chunk_index, v "
            "FROM gateup_fused WHERE flag = 'G'"
        )
        conn.execute(
            "CREATE TEMP TABLE u_fus AS "
            "SELECT row_index, chunk_index, v "
            "FROM gateup_fused WHERE flag = 'U'"
        )
        g_fus = read_2d(conn, "g_fus", SEQ_LEN, FFN_DIM, CHUNK_SIZE)
        u_fus = read_2d(conn, "u_fus", SEQ_LEN, FFN_DIM, CHUNK_SIZE)

        np.testing.assert_allclose(g_fus, g_sep, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(u_fus, u_sep, atol=ATOL, rtol=RTOL)


# ===========================================================================
# §4.3  ROW2COL Pivoting tests
# ===========================================================================

class TestPivotedMatMul:
    """Pivoted MatMul must produce identical results to standard MatMul."""

    def test_square_matmul(self, conn):
        """hidden_dim × hidden_dim — same dims as Q/K/V/O projections."""
        act = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
        weight = rng.standard_normal((HIDDEN_DIM, HIDDEN_DIM)).astype(np.float32)

        load_2d(conn, "act", act, CHUNK_SIZE)
        load_2d(conn, "wt", weight, CHUNK_SIZE)

        # Baseline
        run_steps(conn, matmul_sql("act", "wt", "base", CHUNK_SIZE))
        base = read_2d(conn, "base", SEQ_LEN, HIDDEN_DIM, CHUNK_SIZE)

        # Pivoted (all chunks at once, 1 col per subquery)
        n_chunks = HIDDEN_DIM // CHUNK_SIZE
        run_steps(conn, pivoted_matmul_sql(
            "act", "wt", "piv", n_chunks, CHUNK_SIZE))
        piv = read_2d(conn, "piv", SEQ_LEN, HIDDEN_DIM, CHUNK_SIZE)

        np.testing.assert_allclose(piv, base, atol=ATOL, rtol=RTOL)

    def test_rect_expand(self, conn):
        """hidden_dim → ffn_dim — same shape as gate/up projections."""
        act = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
        weight = rng.standard_normal((FFN_DIM, HIDDEN_DIM)).astype(np.float32)

        load_2d(conn, "act", act, CHUNK_SIZE)
        load_2d(conn, "wt", weight, CHUNK_SIZE)

        run_steps(conn, matmul_sql("act", "wt", "base", CHUNK_SIZE))
        base = read_2d(conn, "base", SEQ_LEN, FFN_DIM, CHUNK_SIZE)

        n_chunks = HIDDEN_DIM // CHUNK_SIZE
        run_steps(conn, pivoted_matmul_sql(
            "act", "wt", "piv", n_chunks, CHUNK_SIZE))
        piv = read_2d(conn, "piv", SEQ_LEN, FFN_DIM, CHUNK_SIZE)

        np.testing.assert_allclose(piv, base, atol=ATOL, rtol=RTOL)

    def test_rect_contract(self, conn):
        """ffn_dim → hidden_dim — same shape as down projection."""
        act = rng.standard_normal((SEQ_LEN, FFN_DIM)).astype(np.float32)
        weight = rng.standard_normal((HIDDEN_DIM, FFN_DIM)).astype(np.float32)

        load_2d(conn, "act", act, CHUNK_SIZE)
        load_2d(conn, "wt", weight, CHUNK_SIZE)

        run_steps(conn, matmul_sql("act", "wt", "base", CHUNK_SIZE))
        base = read_2d(conn, "base", SEQ_LEN, HIDDEN_DIM, CHUNK_SIZE)

        n_chunks = FFN_DIM // CHUNK_SIZE
        run_steps(conn, pivoted_matmul_sql(
            "act", "wt", "piv", n_chunks, CHUNK_SIZE))
        piv = read_2d(conn, "piv", SEQ_LEN, HIDDEN_DIM, CHUNK_SIZE)

        np.testing.assert_allclose(piv, base, atol=ATOL, rtol=RTOL)

    def test_grouped_pivot(self, conn):
        """Test with pivot_width < n_chunks (multiple pivot groups)."""
        act = rng.standard_normal((SEQ_LEN, FFN_DIM)).astype(np.float32)
        weight = rng.standard_normal((HIDDEN_DIM, FFN_DIM)).astype(np.float32)

        load_2d(conn, "act", act, CHUNK_SIZE)
        load_2d(conn, "wt", weight, CHUNK_SIZE)

        run_steps(conn, matmul_sql("act", "wt", "base", CHUNK_SIZE))
        base = read_2d(conn, "base", SEQ_LEN, HIDDEN_DIM, CHUNK_SIZE)

        n_chunks = FFN_DIM // CHUNK_SIZE  # 4
        run_steps(conn, pivoted_matmul_sql(
            "act", "wt", "piv", n_chunks, CHUNK_SIZE,
            pivot_width=2, subquery_width=1))
        piv = read_2d(conn, "piv", SEQ_LEN, HIDDEN_DIM, CHUNK_SIZE)

        np.testing.assert_allclose(piv, base, atol=ATOL, rtol=RTOL)


# ===========================================================================
# §4.1  CTE Merging tests
# ===========================================================================

class TestCTEMerge:
    """CTE merging must produce identical final output to baseline."""

    def test_full_layer_cte_only(self, conn):
        """CTE merge only (no fusion, no pivoting) matches baseline.

        Compares full-pipeline logits because in 1-layer config l0_x_out
        is NOT shared and gets CTE-merged into the logits block.
        """
        config = make_config()
        dag = TensorComputeDAG.build_llama3_8b(config)

        x_in = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
        load_2d(conn, "x_0", x_in, CHUNK_SIZE)
        load_layer_weights(conn, rng)

        # Load final_norm and lm_head for full pipeline
        final_norm = rng.standard_normal(HIDDEN_DIM).astype(np.float32) + 1.0
        lm_head = rng.standard_normal((config.vocab_size, HIDDEN_DIM)).astype(np.float32) * 0.1
        load_norm_weight(conn, "final_norm", final_norm, CHUNK_SIZE)
        load_2d(conn, "lm_head", lm_head, CHUNK_SIZE)

        # Baseline (no optimisations) — skip embed, x_0 already loaded
        baseline_steps = dag_to_sql(dag)
        baseline_no_embed = [(s, n) for s, n in baseline_steps if n != "x_0"]
        run_steps(conn, baseline_no_embed)
        baseline_out = read_2d(conn, "logits", SEQ_LEN, config.vocab_size, CHUNK_SIZE)

        # Drop baseline temps
        for _, name in baseline_no_embed:
            conn.execute(f"DROP TABLE IF EXISTS {name}")

        # CTE-merged (cte_merge only) — skip embed
        opts = PostOptOptions(cte_merge=True, table_fusion=False,
                              row2col_pivot=False)
        cte_steps = postopt_dag_to_sql(dag, opts)
        cte_no_embed = [(s, n) for s, n in cte_steps if n != "x_0"]
        run_steps(conn, cte_no_embed)
        cte_out = read_2d(conn, "logits", SEQ_LEN, config.vocab_size, CHUNK_SIZE)

        np.testing.assert_allclose(cte_out, baseline_out, atol=ATOL, rtol=RTOL)

    def test_intermediate_count(self):
        """Paper §4.1 claims CTE merging reduces per-layer intermediates.

        Baseline: each SQL step creates one temp table.
        CTE-merged: shared nodes + force-materialised cross-epoch nodes.

        Paper's "38 to 7" claim assumes ALL optimisations (fusion + CTE merge).
        With CTE merge ONLY (no fusion), Q and K must be force-materialised
        because their consumers (RoPE_Q, RoPE_K) are in a different CTE epoch.

        CTE-only materialised tables per layer:
          norm1(shared), Q(force), K(force), V(shared),
          q_rope(shared), k_rope(shared), x_after_attn(shared),
          norm2(shared)  = 8
        (x_out is NOT shared for last layer in 1-layer config)
        """
        config = make_config()
        dag = TensorComputeDAG.build_llama3_8b(config)

        # Count baseline steps for layer 0
        baseline_steps = dag_to_sql(dag)
        baseline_layer = [n for _, n in baseline_steps if n.startswith("l0_")]

        # Count CTE-merged steps (no fusion, no pivoting)
        opts = PostOptOptions(cte_merge=True, table_fusion=False,
                              row2col_pivot=False)
        cte_steps = postopt_dag_to_sql(dag, opts)
        cte_layer = [n for _, n in cte_steps if n.startswith("l0_")]

        # Baseline should have many more steps than CTE-merged
        assert len(baseline_layer) > len(cte_layer), (
            f"CTE merge should reduce step count: "
            f"baseline={len(baseline_layer)}, cte={len(cte_layer)}"
        )

    def test_intermediate_count_with_fusion(self):
        """Paper §4.1: CTE merge + fusion → 7 materialised per layer.

        Single-table fusion (paper §4.2) produces one fused table per
        projection group.  Downstream ops filter by flag via cheap CTEs.

        The 7 materialised tables per layer:
          1. norm1_out      (shared: feeds fused QKV)
          2. qkv_fused      (shared: consumed by q_rope, k_rope, attn_vmul)
          3. q_rope         (shared: consumed by qk_scores)
          4. k_rope         (shared: consumed by qk_scores)
          5. x_after_attn   (shared: feeds norm2 + residual_add_2)
          6. norm2_out      (shared: feeds fused gate+up)
          7. x_out          (shared: feeds next layer)

        gate+up fusion CTE-merges entirely into x_out's block (same epoch
        for all consumers), contributing 0 extra tables.
        """
        config = make_config()

        # Use 2 layers so x_out of layer 0 is shared
        config2 = ModelConfig(
            hidden_dim=HIDDEN_DIM, num_q_heads=NUM_Q_HEADS,
            num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
            ffn_dim=FFN_DIM, num_layers=2, vocab_size=16,
            rms_norm_eps=EPS, rope_theta=10000.0,
            max_seq_len=SEQ_LEN, chunk_size=CHUNK_SIZE,
        )
        dag2 = TensorComputeDAG.build_llama3_8b(config2)

        opts = PostOptOptions(cte_merge=True, table_fusion=True,
                              row2col_pivot=False)
        cte_steps = postopt_dag_to_sql(dag2, opts)

        # Count materialised tables for layer 0 only
        l0_tables = [n for _, n in cte_steps if n.startswith("l0_")]
        assert len(l0_tables) == 7, (
            f"Expected 7 materialised per layer (§4.1), got {len(l0_tables)}: "
            f"{l0_tables}"
        )
