"""
Unit tests for TranSQL+ SQL template operators.

Each test creates tiny synthetic data, loads it into an in-memory DuckDB,
executes the SQL from transql_plus.sql_templates, and compares to a NumPy
reference.

Test parameters (must stay consistent):
    CHUNK_SIZE=2, HIDDEN_DIM=4, KV_DIM=2, FFN_DIM=8
    NUM_Q_HEADS=2, NUM_KV_HEADS=1, HEAD_DIM=2, SEQ_LEN=3

Adapted from: AQP_middleware/transql/python/unit_tests/test_sql_templates.py
Key change: chunk_index uses raw offset (0, 2, 4, ...) per Decision D7.
"""

import numpy as np
import pytest

from tests.conftest import (
    load_2d, load_norm_weight, load_rope_table, read_2d, run_steps,
)
from transql_plus.sql_templates import (
    embed_lookup_sql, matmul_sql, rmsnorm_sql, rope_sql,
    qk_attn_sql, softmax_sql, attn_vmul_sql, swiglu_sql, residual_add_sql,
)

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
CHUNK_SIZE   = 2
SEQ_LEN      = 3
HIDDEN_DIM   = 4      # 2 chunks per token row
KV_DIM       = 2      # 1 chunk; num_kv_heads=1, head_dim=2
FFN_DIM      = 8      # 4 chunks
VOCAB_SIZE   = 16
NUM_Q_HEADS  = 2
NUM_KV_HEADS = 1
HEAD_DIM     = 2      # hidden_dim / num_q_heads
EPS          = 1e-5
ATOL         = 1e-4
RTOL         = 1e-4

rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Extra DuckDB helpers for test data
# ---------------------------------------------------------------------------

def load_rope_out(con, name, matrix, chunk_size=CHUNK_SIZE):
    """Load [seq, dim] matrix in RoPE split output format:
    (row_index, chunk_index, v_even FLOAT[], v_odd FLOAT[])."""
    half = chunk_size // 2
    n, dim = matrix.shape
    con.execute(
        f"CREATE TEMP TABLE {name} "
        f"(row_index INTEGER, chunk_index INTEGER, "
        f"v_even FLOAT[], v_odd FLOAT[])"
    )
    rows = []
    n_chunks = dim // chunk_size
    for r in range(n):
        for c in range(n_chunks):
            offset = c * chunk_size   # raw offset (D7)
            v_even = [float(matrix[r, c * chunk_size + 2 * i])
                      for i in range(half)]
            v_odd = [float(matrix[r, c * chunk_size + 2 * i + 1])
                     for i in range(half)]
            rows.append((r, offset, v_even, v_odd))
    con.executemany(f"INSERT INTO {name} VALUES (?, ?, ?, ?)", rows)


def load_scores(con, name, scores_np):
    """Load [q_tok, k_tok, heads] score array as scalar rows."""
    con.execute(
        f"CREATE TEMP TABLE {name} "
        f"(q_tok INTEGER, k_tok INTEGER, head_id INTEGER, score FLOAT)"
    )
    q_len, k_len, n_heads = scores_np.shape
    rows = [(q, k, h, float(scores_np[q, k, h]))
            for q in range(q_len) for k in range(k_len)
            for h in range(n_heads)]
    con.executemany(f"INSERT INTO {name} VALUES (?, ?, ?, ?)", rows)


def load_attn_weights(con, name, attn_np):
    """Load [q_tok, k_tok, heads] attention weights as scalar rows."""
    con.execute(
        f"CREATE TEMP TABLE {name} "
        f"(q_tok INTEGER, k_tok INTEGER, head_id INTEGER, attn_weight FLOAT)"
    )
    q_len, k_len, n_heads = attn_np.shape
    rows = [(q, k, h, float(attn_np[q, k, h]))
            for q in range(q_len) for k in range(k_len)
            for h in range(n_heads)]
    con.executemany(f"INSERT INTO {name} VALUES (?, ?, ?, ?)", rows)


def read_rope_out(con, name, n_rows, dim, chunk_size=CHUNK_SIZE):
    """Read RoPE split output back to interleaved [n_rows, dim]."""
    half = chunk_size // 2
    rows = con.execute(
        f"SELECT row_index, chunk_index, v_even, v_odd FROM {name} "
        f"ORDER BY row_index, chunk_index"
    ).fetchall()
    out = np.zeros((n_rows, dim), dtype=np.float32)
    for row_idx, chunk_idx, v_even, v_odd in rows:
        col_start = chunk_idx   # raw offset (D7)
        for i in range(half):
            out[row_idx, col_start + 2 * i] = v_even[i]
            out[row_idx, col_start + 2 * i + 1] = v_odd[i]
    return out


def read_scores(con, name, seq_len, num_heads, val_col="score"):
    """Read scalar score table into [q, k, heads] array."""
    rows = con.execute(
        f"SELECT q_tok, k_tok, head_id, {val_col} FROM {name} "
        f"ORDER BY head_id, q_tok, k_tok"
    ).fetchall()
    out = np.zeros((seq_len, seq_len, num_heads), dtype=np.float32)
    for q_tok, k_tok, head_id, val in rows:
        out[int(q_tok), int(k_tok), int(head_id)] = val
    return out


# ---------------------------------------------------------------------------
# NumPy reference implementations
# ---------------------------------------------------------------------------

def ref_embed_lookup(embed, token_ids):
    return embed[token_ids].astype(np.float32)


def ref_matmul(act, weight):
    return (act @ weight.T).astype(np.float32)


def ref_rmsnorm(x, gamma, hidden_dim, eps):
    sq_sum = np.sum(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(sq_sum / hidden_dim + eps) * gamma).astype(np.float32)


def ref_rope(q, cos, sin, chunk_size=CHUNK_SIZE):
    half = chunk_size // 2
    num_chunks = q.shape[1] // chunk_size
    out = q.copy().astype(np.float32)
    for pos in range(q.shape[0]):
        for c in range(num_chunks):
            for i in range(half):
                q_e = np.float32(q[pos, c * chunk_size + 2 * i])
                q_o = np.float32(q[pos, c * chunk_size + 2 * i + 1])
                c_i = np.float32(cos[pos, c, i])
                s_i = np.float32(sin[pos, c, i])
                out[pos, c * chunk_size + 2 * i] = q_e * c_i - q_o * s_i
                out[pos, c * chunk_size + 2 * i + 1] = q_o * c_i + q_e * s_i
    return out


def ref_qk_attn(q_rot, k_rot, num_q_heads, num_kv_heads, head_dim):
    seq = q_rot.shape[0]
    gs = num_q_heads // num_kv_heads
    out = np.zeros((seq, seq, num_q_heads), dtype=np.float32)
    for h in range(num_q_heads):
        kv_h = h // gs
        q_h = q_rot[:, h * head_dim:(h + 1) * head_dim]
        k_h = k_rot[:, kv_h * head_dim:(kv_h + 1) * head_dim]
        out[:, :, h] = (q_h @ k_h.T).astype(np.float32)
    return out


def ref_softmax(scores):
    s_max = np.max(scores, axis=1, keepdims=True)
    exp_s = np.exp(scores - s_max).astype(np.float32)
    return (exp_s / exp_s.sum(axis=1, keepdims=True)).astype(np.float32)


def ref_attn_vmul(attn, v, num_q_heads, num_kv_heads, head_dim):
    seq = attn.shape[0]
    gs = num_q_heads // num_kv_heads
    out = np.zeros((seq, num_q_heads * head_dim), dtype=np.float32)
    for h in range(num_q_heads):
        kv_h = h // gs
        v_h = v[:, kv_h * head_dim:(kv_h + 1) * head_dim]
        out[:, h * head_dim:(h + 1) * head_dim] = (attn[:, :, h] @ v_h).astype(np.float32)
    return out


def ref_swiglu(gate, up):
    silu_gate = gate / (1.0 + np.exp(-gate.astype(np.float64)))
    return (silu_gate * up).astype(np.float32)


def ref_residual_add(a, b):
    return (a + b).astype(np.float32)


# ===========================================================================
# Test cases
# ===========================================================================

class TestEmbedLookup:
    def test_basic(self, conn):
        embed = rng.standard_normal((VOCAB_SIZE, HIDDEN_DIM)).astype(np.float32)
        token_ids = rng.integers(0, VOCAB_SIZE, size=SEQ_LEN)

        conn.execute("CREATE TEMP TABLE tok (pos INTEGER, token_id INTEGER)")
        conn.executemany("INSERT INTO tok VALUES (?, ?)",
                         [(i, int(token_ids[i])) for i in range(SEQ_LEN)])
        load_2d(conn, "embed", embed, CHUNK_SIZE)
        run_steps(conn, embed_lookup_sql("tok", "embed", "out"))

        result = read_2d(conn, "out", SEQ_LEN, HIDDEN_DIM, CHUNK_SIZE)
        expected = ref_embed_lookup(embed, token_ids)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


class TestMatMul:
    def test_square(self, conn):
        act = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
        weight = rng.standard_normal((HIDDEN_DIM, HIDDEN_DIM)).astype(np.float32)

        load_2d(conn, "act", act, CHUNK_SIZE)
        load_2d(conn, "w", weight, CHUNK_SIZE)
        run_steps(conn, matmul_sql("act", "w", "out", CHUNK_SIZE))

        result = read_2d(conn, "out", SEQ_LEN, HIDDEN_DIM, CHUNK_SIZE)
        expected = ref_matmul(act, weight)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_rect_expand(self, conn):
        act = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
        weight = rng.standard_normal((FFN_DIM, HIDDEN_DIM)).astype(np.float32)

        load_2d(conn, "act", act, CHUNK_SIZE)
        load_2d(conn, "w", weight, CHUNK_SIZE)
        run_steps(conn, matmul_sql("act", "w", "out", CHUNK_SIZE))

        result = read_2d(conn, "out", SEQ_LEN, FFN_DIM, CHUNK_SIZE)
        expected = ref_matmul(act, weight)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_rect_contract(self, conn):
        act = rng.standard_normal((SEQ_LEN, FFN_DIM)).astype(np.float32)
        weight = rng.standard_normal((HIDDEN_DIM, FFN_DIM)).astype(np.float32)

        load_2d(conn, "act", act, CHUNK_SIZE)
        load_2d(conn, "w", weight, CHUNK_SIZE)
        run_steps(conn, matmul_sql("act", "w", "out", CHUNK_SIZE))

        result = read_2d(conn, "out", SEQ_LEN, HIDDEN_DIM, CHUNK_SIZE)
        expected = ref_matmul(act, weight)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


class TestRMSNorm:
    def test_basic(self, conn):
        x = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
        gamma = (rng.standard_normal(HIDDEN_DIM) + 1.0).astype(np.float32)

        load_2d(conn, "inp", x, CHUNK_SIZE)
        load_norm_weight(conn, "gamma", gamma, CHUNK_SIZE)
        run_steps(conn, rmsnorm_sql("inp", "gamma", "out", HIDDEN_DIM, EPS, CHUNK_SIZE))

        result = read_2d(conn, "out", SEQ_LEN, HIDDEN_DIM, CHUNK_SIZE)
        expected = ref_rmsnorm(x, gamma, HIDDEN_DIM, EPS)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_unit_gamma(self, conn):
        x = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
        gamma = np.ones(HIDDEN_DIM, dtype=np.float32)

        load_2d(conn, "inp", x, CHUNK_SIZE)
        load_norm_weight(conn, "gamma", gamma, CHUNK_SIZE)
        run_steps(conn, rmsnorm_sql("inp", "gamma", "out", HIDDEN_DIM, EPS, CHUNK_SIZE))

        result = read_2d(conn, "out", SEQ_LEN, HIDDEN_DIM, CHUNK_SIZE)
        rms = np.sqrt(np.mean(result.astype(np.float64) ** 2, axis=-1))
        np.testing.assert_allclose(rms, np.ones(SEQ_LEN), atol=2e-4)


class TestRoPE:
    @pytest.fixture(autouse=True)
    def _setup_rope(self):
        num_chunks = HIDDEN_DIM // CHUNK_SIZE
        half = CHUNK_SIZE // 2
        self.cos = rng.uniform(0.5, 1.0, (SEQ_LEN, num_chunks, half)).astype(np.float32)
        self.sin = rng.uniform(0.0, 0.5, (SEQ_LEN, num_chunks, half)).astype(np.float32)

    def test_basic(self, conn):
        q = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)

        load_2d(conn, "q", q, CHUNK_SIZE)
        load_rope_table(conn, self.cos, self.sin, CHUNK_SIZE)
        run_steps(conn, rope_sql("q", "rope", "out", CHUNK_SIZE))

        result = read_rope_out(conn, "out", SEQ_LEN, HIDDEN_DIM, CHUNK_SIZE)
        expected = ref_rope(q, self.cos, self.sin, CHUNK_SIZE)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_identity_at_zero_angle(self, conn):
        q = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
        num_chunks = HIDDEN_DIM // CHUNK_SIZE
        half = CHUNK_SIZE // 2
        cos = np.ones((SEQ_LEN, num_chunks, half), dtype=np.float32)
        sin = np.zeros((SEQ_LEN, num_chunks, half), dtype=np.float32)

        load_2d(conn, "q", q, CHUNK_SIZE)
        load_rope_table(conn, cos, sin, CHUNK_SIZE)
        run_steps(conn, rope_sql("q", "rope", "out", CHUNK_SIZE))

        result = read_rope_out(conn, "out", SEQ_LEN, HIDDEN_DIM, CHUNK_SIZE)
        np.testing.assert_allclose(result, q, atol=ATOL, rtol=RTOL)


class TestQKAttn:
    def test_basic(self, conn):
        q_rot = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
        k_rot = rng.standard_normal((SEQ_LEN, KV_DIM)).astype(np.float32)

        load_rope_out(conn, "q_rope", q_rot, CHUNK_SIZE)
        load_rope_out(conn, "k_rope", k_rot, CHUNK_SIZE)
        run_steps(conn, qk_attn_sql("q_rope", "k_rope", "scores",
                                     NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, CHUNK_SIZE))

        result = read_scores(conn, "scores", SEQ_LEN, NUM_Q_HEADS, "score")
        expected = ref_qk_attn(q_rot, k_rot, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)
        lower = np.tril(np.ones((SEQ_LEN, SEQ_LEN), dtype=bool))
        np.testing.assert_allclose(result[lower], expected[lower], atol=ATOL, rtol=RTOL)


class TestSoftmax:
    def test_basic(self, conn):
        scores_np = rng.standard_normal((SEQ_LEN, SEQ_LEN, NUM_Q_HEADS)).astype(np.float32)

        load_scores(conn, "scores_in", scores_np)
        run_steps(conn, softmax_sql("scores_in", "attn_w"))

        result = read_scores(conn, "attn_w", SEQ_LEN, NUM_Q_HEADS, "attn_weight")
        expected = ref_softmax(scores_np)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_rows_sum_to_one(self, conn):
        scores_np = rng.standard_normal((SEQ_LEN, SEQ_LEN, NUM_Q_HEADS)).astype(np.float32)

        load_scores(conn, "scores_in", scores_np)
        run_steps(conn, softmax_sql("scores_in", "attn_w"))

        result = read_scores(conn, "attn_w", SEQ_LEN, NUM_Q_HEADS, "attn_weight")
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-5)

    def test_stable_variant(self, conn):
        scores_np = rng.standard_normal((SEQ_LEN, SEQ_LEN, NUM_Q_HEADS)).astype(np.float32)

        load_scores(conn, "scores_in", scores_np)
        run_steps(conn, softmax_sql("scores_in", "attn_w", stable=True))

        result = read_scores(conn, "attn_w", SEQ_LEN, NUM_Q_HEADS, "attn_weight")
        expected = ref_softmax(scores_np)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


class TestAttnVMul:
    def test_basic(self, conn):
        attn_np = rng.random((SEQ_LEN, SEQ_LEN, NUM_Q_HEADS)).astype(np.float32)
        attn_np /= attn_np.sum(axis=1, keepdims=True)
        v_np = rng.standard_normal((SEQ_LEN, KV_DIM)).astype(np.float32)

        load_attn_weights(conn, "attn_w", attn_np)
        load_2d(conn, "v", v_np, CHUNK_SIZE)
        run_steps(conn, attn_vmul_sql("attn_w", "v", "attn_out",
                                       NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, CHUNK_SIZE))

        result = read_2d(conn, "attn_out", SEQ_LEN, HIDDEN_DIM, CHUNK_SIZE)
        expected = ref_attn_vmul(attn_np, v_np, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


class TestSwiGLU:
    def test_basic(self, conn):
        gate_np = rng.standard_normal((SEQ_LEN, FFN_DIM)).astype(np.float32)
        up_np = rng.standard_normal((SEQ_LEN, FFN_DIM)).astype(np.float32)

        load_2d(conn, "gate", gate_np, CHUNK_SIZE)
        load_2d(conn, "up", up_np, CHUNK_SIZE)
        run_steps(conn, swiglu_sql("gate", "up", "out"))

        result = read_2d(conn, "out", SEQ_LEN, FFN_DIM, CHUNK_SIZE)
        expected = ref_swiglu(gate_np, up_np)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


class TestResidualAdd:
    def test_basic(self, conn):
        a = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
        b = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)

        load_2d(conn, "ta", a, CHUNK_SIZE)
        load_2d(conn, "tb", b, CHUNK_SIZE)
        run_steps(conn, residual_add_sql("ta", "tb", "out"))

        result = read_2d(conn, "out", SEQ_LEN, HIDDEN_DIM, CHUNK_SIZE)
        expected = ref_residual_add(a, b)
        np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)

    def test_zero_addend(self, conn):
        a = rng.standard_normal((SEQ_LEN, HIDDEN_DIM)).astype(np.float32)
        b = np.zeros((SEQ_LEN, HIDDEN_DIM), dtype=np.float32)

        load_2d(conn, "ta", a, CHUNK_SIZE)
        load_2d(conn, "tb", b, CHUNK_SIZE)
        run_steps(conn, residual_add_sql("ta", "tb", "out"))

        result = read_2d(conn, "out", SEQ_LEN, HIDDEN_DIM, CHUNK_SIZE)
        np.testing.assert_allclose(result, a, atol=ATOL, rtol=RTOL)
