"""
End-to-end test for one Llama3-style transformer layer.

Runs the full attention + FFN pipeline through DuckDB SQL (via dag_to_sql)
and compares against a NumPy reference.

Uses the same tiny dimensions as test_sql_templates.py:
    CHUNK_SIZE=2, HIDDEN_DIM=4, KV_DIM=2, FFN_DIM=8
    NUM_Q_HEADS=2, NUM_KV_HEADS=1, HEAD_DIM=2, SEQ_LEN=3

Source: AQP_middleware/transql/python/unit_tests/test_single_layer.py
"""

import numpy as np
import pytest

from tests.conftest import (
    load_2d, load_norm_weight, load_rope_table, read_2d, run_steps,
)
from transql_plus.config import ModelConfig
from transql_plus.compute_dag import TensorComputeDAG, TensorOpType
from transql_plus.dag_to_sql import expand_node, dag_to_sql

# ---------------------------------------------------------------------------
# Test parameters (must match test_sql_templates.py)
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

rng = np.random.default_rng(123)


# ---------------------------------------------------------------------------
# Build a 1-layer config and DAG
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# NumPy reference for one full layer
# ---------------------------------------------------------------------------

def np_rmsnorm(x, gamma, eps):
    sq = np.sum(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(sq / x.shape[-1] + eps) * gamma).astype(np.float32)


def np_rope(q, cos_arr, sin_arr, chunk_size):
    """Apply RoPE using even/odd split convention."""
    half = chunk_size // 2
    n_chunks = q.shape[1] // chunk_size
    out = np.zeros_like(q, dtype=np.float32)
    for pos in range(q.shape[0]):
        for c in range(n_chunks):
            for i in range(half):
                e = q[pos, c * chunk_size + 2 * i]
                o = q[pos, c * chunk_size + 2 * i + 1]
                ci = cos_arr[pos, c, i]
                si = sin_arr[pos, c, i]
                out[pos, c * chunk_size + 2 * i] = e * ci - o * si
                out[pos, c * chunk_size + 2 * i + 1] = o * ci + e * si
    return out


def np_qk_attn(q_rot, k_rot, num_q_heads, num_kv_heads, head_dim):
    """QK attention with GQA + causal mask. Returns [seq, seq, heads]."""
    seq = q_rot.shape[0]
    gs = num_q_heads // num_kv_heads
    scores = np.full((seq, seq, num_q_heads), -1e30, dtype=np.float32)
    for h in range(num_q_heads):
        kv_h = h // gs
        q_h = q_rot[:, h * head_dim:(h + 1) * head_dim]
        k_h = k_rot[:, kv_h * head_dim:(kv_h + 1) * head_dim]
        raw = (q_h @ k_h.T).astype(np.float32)
        # causal mask
        for q_tok in range(seq):
            for k_tok in range(seq):
                if k_tok <= q_tok:
                    scores[q_tok, k_tok, h] = raw[q_tok, k_tok]
    return scores


def np_softmax_causal(scores):
    """Softmax over k dimension, ignoring masked (== -1e30) positions."""
    out = np.zeros_like(scores)
    for q in range(scores.shape[0]):
        for h in range(scores.shape[2]):
            row = scores[q, :, h]
            valid = row > -1e20   # only causal entries
            exp_vals = np.exp(row[valid] - np.max(row[valid]))
            exp_vals /= exp_vals.sum()
            out[q, valid, h] = exp_vals
    return out.astype(np.float32)


def np_attn_vmul(attn_w, v, num_q_heads, num_kv_heads, head_dim):
    """Attention-weighted value sum with GQA."""
    seq = attn_w.shape[0]
    gs = num_q_heads // num_kv_heads
    out = np.zeros((seq, num_q_heads * head_dim), dtype=np.float32)
    for h in range(num_q_heads):
        kv_h = h // gs
        v_h = v[:, kv_h * head_dim:(kv_h + 1) * head_dim]
        out[:, h * head_dim:(h + 1) * head_dim] = (attn_w[:, :, h] @ v_h)
    return out.astype(np.float32)


def np_swiglu(gate, up):
    silu = gate / (1.0 + np.exp(-gate.astype(np.float64)))
    return (silu * up).astype(np.float32)


def numpy_single_layer(x_in, weights, cos_arr, sin_arr):
    """Full NumPy reference for one transformer layer."""
    # Pre-attention RMSNorm
    norm1 = np_rmsnorm(x_in, weights["norm1"], EPS)

    # Q/K/V projections
    q = (norm1 @ weights["q_proj"].T).astype(np.float32)
    k = (norm1 @ weights["k_proj"].T).astype(np.float32)
    v = (norm1 @ weights["v_proj"].T).astype(np.float32)

    # RoPE
    q_rope = np_rope(q, cos_arr, sin_arr, CHUNK_SIZE)
    k_rope = np_rope(k, cos_arr[:, :KV_DIM // CHUNK_SIZE, :],
                     sin_arr[:, :KV_DIM // CHUNK_SIZE, :], CHUNK_SIZE)

    # QK attention + softmax
    scores = np_qk_attn(q_rope, k_rope, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)
    attn_w = np_softmax_causal(scores)

    # Attention x V
    attn_out = np_attn_vmul(attn_w, v, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)

    # O projection + residual
    o = (attn_out @ weights["o_proj"].T).astype(np.float32)
    x_after_attn = x_in + o

    # Pre-FFN RMSNorm
    norm2 = np_rmsnorm(x_after_attn, weights["norm2"], EPS)

    # Gate/Up → SwiGLU → Down + residual
    gate = (norm2 @ weights["gate_proj"].T).astype(np.float32)
    up = (norm2 @ weights["up_proj"].T).astype(np.float32)
    ffn_act = np_swiglu(gate, up)
    down = (ffn_act @ weights["down_proj"].T).astype(np.float32)
    x_out = x_after_attn + down

    return x_out


# ---------------------------------------------------------------------------
# Load weight tables into DuckDB
# ---------------------------------------------------------------------------

def load_layer_weights(conn, weights, cos_arr, sin_arr):
    """Load all layer-0 weight tables + RoPE into DuckDB."""
    # 2D weight matrices
    for name in ("q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"):
        load_2d(conn, f"layer_0_{name}", weights[name], CHUNK_SIZE)

    # 1D norm weights (Decision D1: no row_index)
    load_norm_weight(conn, "layer_0_norm1", weights["norm1"], CHUNK_SIZE)
    load_norm_weight(conn, "layer_0_norm2", weights["norm2"], CHUNK_SIZE)

    # RoPE cos/sin table
    load_rope_table(conn, cos_arr, sin_arr, CHUNK_SIZE, "rope")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestSingleLayer:
    """Run one full Llama3 layer through SQL and compare to NumPy."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Generate synthetic weights and input."""
        self.x_in = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)

        self.weights = {
            "norm1":     rng.standard_normal(HIDDEN_DIM).astype(np.float32) + 1.0,
            "q_proj":    rng.standard_normal((HIDDEN_DIM, HIDDEN_DIM)).astype(np.float32) * 0.1,
            "k_proj":    rng.standard_normal((KV_DIM, HIDDEN_DIM)).astype(np.float32) * 0.1,
            "v_proj":    rng.standard_normal((KV_DIM, HIDDEN_DIM)).astype(np.float32) * 0.1,
            "o_proj":    rng.standard_normal((HIDDEN_DIM, HIDDEN_DIM)).astype(np.float32) * 0.1,
            "norm2":     rng.standard_normal(HIDDEN_DIM).astype(np.float32) + 1.0,
            "gate_proj": rng.standard_normal((FFN_DIM, HIDDEN_DIM)).astype(np.float32) * 0.1,
            "up_proj":   rng.standard_normal((FFN_DIM, HIDDEN_DIM)).astype(np.float32) * 0.1,
            "down_proj": rng.standard_normal((HIDDEN_DIM, FFN_DIM)).astype(np.float32) * 0.1,
        }

        half = CHUNK_SIZE // 2
        n_chunks = HIDDEN_DIM // CHUNK_SIZE
        self.cos_arr = rng.uniform(0.5, 1.0, (SEQ_LEN, n_chunks, half)).astype(np.float32)
        self.sin_arr = rng.uniform(0.0, 0.5, (SEQ_LEN, n_chunks, half)).astype(np.float32)

    def test_full_layer_via_dag(self, conn):
        """Build a 1-layer DAG, expand to SQL, execute, compare to NumPy."""
        config = make_config()
        dag = TensorComputeDAG.build_llama3_8b(config)

        # Load input activation as x_0 (embedding output)
        load_2d(conn, "x_0", self.x_in, CHUNK_SIZE)
        load_layer_weights(conn, self.weights, self.cos_arr, self.sin_arr)

        # Expand DAG to SQL and run.
        # Skip embed step (x_0 pre-loaded) and post-layer steps
        # (final_norm, logits) which need weight tables we don't load.
        steps = dag_to_sql(dag)
        layer_steps = [(sql, name) for sql, name in steps
                       if name.startswith("l0_")]
        run_steps(conn, layer_steps)

        result = read_2d(conn, "l0_x_out", SEQ_LEN, HIDDEN_DIM, CHUNK_SIZE)
        expected = numpy_single_layer(self.x_in, self.weights,
                                      self.cos_arr, self.sin_arr)

        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_attention_block(self, conn):
        """Test just the attention sub-block (pre-norm through residual add)."""
        load_2d(conn, "x_in", self.x_in, CHUNK_SIZE)
        load_layer_weights(conn, self.weights, self.cos_arr, self.sin_arr)

        # Manually run attention block steps
        from transql_plus.sql_templates import (
            rmsnorm_sql, matmul_sql, rope_sql, qk_attn_sql,
            softmax_sql, attn_vmul_sql, residual_add_sql,
        )

        all_steps = []
        all_steps.extend(rmsnorm_sql("x_in", "layer_0_norm1", "norm1_out",
                                     HIDDEN_DIM, EPS, CHUNK_SIZE))
        all_steps.extend(matmul_sql("norm1_out", "layer_0_q_proj", "q", CHUNK_SIZE))
        all_steps.extend(matmul_sql("norm1_out", "layer_0_k_proj", "k", CHUNK_SIZE))
        all_steps.extend(matmul_sql("norm1_out", "layer_0_v_proj", "v", CHUNK_SIZE))
        all_steps.extend(rope_sql("q", "rope", "q_rope", CHUNK_SIZE))
        all_steps.extend(rope_sql("k", "rope", "k_rope", CHUNK_SIZE))
        all_steps.extend(qk_attn_sql("q_rope", "k_rope", "scores",
                                     NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, CHUNK_SIZE))
        all_steps.extend(softmax_sql("scores", "attn_w"))
        all_steps.extend(attn_vmul_sql("attn_w", "v", "attn_out",
                                       NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, CHUNK_SIZE))
        all_steps.extend(matmul_sql("attn_out", "layer_0_o_proj", "o", CHUNK_SIZE))
        all_steps.extend(residual_add_sql("x_in", "o", "x_after_attn"))

        run_steps(conn, all_steps)

        # NumPy reference for attention block
        norm1 = np_rmsnorm(self.x_in, self.weights["norm1"], EPS)
        q = (norm1 @ self.weights["q_proj"].T).astype(np.float32)
        k = (norm1 @ self.weights["k_proj"].T).astype(np.float32)
        v = (norm1 @ self.weights["v_proj"].T).astype(np.float32)
        q_rope = np_rope(q, self.cos_arr, self.sin_arr, CHUNK_SIZE)
        kv_chunks = KV_DIM // CHUNK_SIZE
        k_rope = np_rope(k, self.cos_arr[:, :kv_chunks, :],
                         self.sin_arr[:, :kv_chunks, :], CHUNK_SIZE)
        scores = np_qk_attn(q_rope, k_rope, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)
        attn_w = np_softmax_causal(scores)
        attn_out = np_attn_vmul(attn_w, v, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)
        o = (attn_out @ self.weights["o_proj"].T).astype(np.float32)
        expected = self.x_in + o

        result = read_2d(conn, "x_after_attn", SEQ_LEN, HIDDEN_DIM, CHUNK_SIZE)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_ffn_block(self, conn):
        """Test just the FFN sub-block (pre-norm through residual add)."""
        # Use x_in directly as FFN input (skip attention for isolation)
        load_2d(conn, "x_attn", self.x_in, CHUNK_SIZE)
        load_layer_weights(conn, self.weights, self.cos_arr, self.sin_arr)

        from transql_plus.sql_templates import (
            rmsnorm_sql, matmul_sql, swiglu_sql, residual_add_sql,
        )

        all_steps = []
        all_steps.extend(rmsnorm_sql("x_attn", "layer_0_norm2", "norm2_out",
                                     HIDDEN_DIM, EPS, CHUNK_SIZE))
        all_steps.extend(matmul_sql("norm2_out", "layer_0_gate_proj", "gate", CHUNK_SIZE))
        all_steps.extend(matmul_sql("norm2_out", "layer_0_up_proj", "up", CHUNK_SIZE))
        all_steps.extend(swiglu_sql("gate", "up", "ffn_act"))
        all_steps.extend(matmul_sql("ffn_act", "layer_0_down_proj", "down", CHUNK_SIZE))
        all_steps.extend(residual_add_sql("x_attn", "down", "x_out"))

        run_steps(conn, all_steps)

        # NumPy reference
        norm2 = np_rmsnorm(self.x_in, self.weights["norm2"], EPS)
        gate = (norm2 @ self.weights["gate_proj"].T).astype(np.float32)
        up = (norm2 @ self.weights["up_proj"].T).astype(np.float32)
        ffn_act = np_swiglu(gate, up)
        down = (ffn_act @ self.weights["down_proj"].T).astype(np.float32)
        expected = self.x_in + down

        result = read_2d(conn, "x_out", SEQ_LEN, HIDDEN_DIM, CHUNK_SIZE)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)
