"""
Template-based SQL code generation for TranSQL+.

Paper reference: Section 3.2, Table 1.

Five core operator categories (Section 3.2.1):
  1. Matrix multiplication  — JOIN on shared dims, SUM(DOT()), GROUP BY free dims
  2. Element-wise functions — apply f() to each chunk independently
  3. Element-wise arithmetic — JOIN on matching indices, apply op
  4. Shape manipulation     — recompute indices via projection
  5. Normalisation          — three-step: apply f, aggregate with agg, post-process with g

Each template function returns list[tuple[str, str]]:
    [(sql_body, output_table_name), ...]

DuckDB function mapping (Decision D3):
    Paper abstract     → DuckDB concrete
    DOT(c1, c2)       → list_dot_product(c1, c2)
    exp(x)            → exp(x)              [standard SQL]
    SUM()             → SUM()               [standard SQL]
    element-wise ops  → list_transform(generate_series(1, len(v)), i -> ...)

Chunk index convention (Decision D7):
    chunk_index uses raw byte offset (0, 32, 64, ...) per Algorithm 1.
    Re-chunk GROUP BY: out_col - (out_col % chunk_size).
"""

from __future__ import annotations

SqlStep = tuple[str, str]       # (sql_body, table_name)
SqlSteps = list[SqlStep]


# ---------------------------------------------------------------------------
# 1. Lookup table  (Section 3.2.2 — Special Cases)
# ---------------------------------------------------------------------------

def embed_lookup_sql(tokens_table: str, embed_table: str,
                     out_table: str) -> SqlSteps:
    """Embedding lookup via equi-join on token id.

    Paper §3.2.2: "Lookup tables … can be naturally expressed as equi-join
    operations on the token index."

    OP_attr: Lookup({tokens, embed}, {pos}, {(token_id, row_index)}, {pos})
      F = {pos}  — each token position is a free dimension in the output
      S = {(token_id, row_index)}  — join key mapping tokens to embedding rows
      G = {pos}  — output grouped by position

    tokens_table schema: (pos INTEGER, token_id INTEGER)
    embed_table  schema: (row_index INTEGER, chunk_index INTEGER, v FLOAT[])
    Output:              (row_index INTEGER, chunk_index INTEGER, v FLOAT[])
                         where row_index = pos
    """
    sql = (
        f"SELECT t.pos AS row_index, e.chunk_index, e.v "
        f"FROM {tokens_table} t "
        f"JOIN {embed_table} e ON t.token_id = e.row_index "
        f"ORDER BY t.pos, e.chunk_index"
    )
    return [(sql, out_table)]


# ---------------------------------------------------------------------------
# 2. Matrix multiplication  (Section 3.2.1)
# ---------------------------------------------------------------------------

def matmul_sql(act_table: str, weight_table: str, out_table: str,
               chunk_size: int = 32) -> SqlSteps:
    """Chunked matrix multiplication.

    Paper §3.2.1, Table 1:
      SELECT A.i, B.j, SUM(DOT(A.w^(c), B.w^(c)))
      FROM A JOIN B ON A.c = B.c
      GROUP BY A.i, B.j

    OP_attr: Matmul({A, B}, emptyset, {(chunk_A, chunk_B)}, {row_A, row_B})
      F = emptyset  — contracted dimension is eliminated
      S = {(chunk_index_A, chunk_index_B)}  — join on chunk alignment
      G = {row_index_A, row_index_B}  — output indexed by (act_row, weight_row)

    Step 1: dot-product intermediates → {out}_dp (act_row, out_col, val)
    Step 2: re-chunk scalars into FLOAT[] → out (row_index, chunk_index, v)
    """
    cs = str(chunk_size)
    dp = out_table + "_dp"

    # DOT() → list_dot_product() (Decision D3)
    step1 = (
        f"SELECT a.row_index AS act_row, w.row_index AS out_col, "
        f"SUM(list_dot_product(a.v, w.v)) AS val "
        f"FROM {act_table} a "
        f"JOIN {weight_table} w ON a.chunk_index = w.chunk_index "
        f"GROUP BY a.row_index, w.row_index"
    )

    # Re-chunk: raw offset chunk_index (Decision D7)
    # out_col - (out_col % chunk_size) gives the raw offset for the chunk group
    step2 = (
        f"SELECT act_row AS row_index, "
        f"out_col - (out_col % {cs}) AS chunk_index, "
        f"array_agg(val ORDER BY out_col) AS v "
        f"FROM {dp} "
        f"GROUP BY act_row, out_col - (out_col % {cs})"
    )

    return [(step1, dp), (step2, out_table)]


# ---------------------------------------------------------------------------
# 3. Normalisation — RMSNorm  (Section 3.2.1)
# ---------------------------------------------------------------------------

def rmsnorm_sql(input_table: str, gamma_table: str, out_table: str,
                hidden_dim: int, eps: float,
                chunk_size: int = 32) -> SqlSteps:
    """RMS normalisation with learnable gamma.

    Paper §3.2.1: Normalization template — Normalize_{f, agg, g}:
      f   = square (element-wise)
      agg = SUM
      g   = x / sqrt(mean_sq + eps) * gamma

    Step 1: sum of squares → {out}_sq
    Step 2: normalise + scale → out

    gamma_table schema: (chunk_index INTEGER, v FLOAT[])  (Decision D1: no row_index)
    """
    sq = out_table + "_sq"

    eps_str = f"{eps:.10f}"

    step1 = (
        f"SELECT row_index, "
        f"SUM(list_sum(list_transform(v, x -> x * x))) AS sq_sum "
        f"FROM {input_table} "
        f"GROUP BY row_index"
    )

    step2 = (
        f"SELECT n.row_index, n.chunk_index, "
        f"list_transform(generate_series(1, len(n.v)), "
        f"i -> CAST(n.v[i] / sqrt(s.sq_sum / {hidden_dim}.0 + {eps_str}) "
        f"* w.v[i] AS FLOAT)) AS v "
        f"FROM {input_table} n "
        f"JOIN {sq} s ON n.row_index = s.row_index "
        f"JOIN {gamma_table} w ON n.chunk_index = w.chunk_index"
    )

    return [(step1, sq), (step2, out_table)]


# ---------------------------------------------------------------------------
# 4. Element-wise function — RoPE  (Decision D2)
# ---------------------------------------------------------------------------

def rope_sql(q_table: str, rope_table: str, out_table: str,
             chunk_size: int = 32) -> SqlSteps:
    """Rotary positional encoding.

    Not in the paper — RoPE is model-specific (Llama architecture).
    Expressible as category 3 (element-wise arithmetic) + category 4
    (shape manipulation) but implemented as a single step for efficiency.
    Decomposition would add 4+ intermediate tables per application.
    See Decision D2.

    Rotation for pair index i (1-based):
      v_even[i] = q[2i-1]*cos[i] - q[2i]*sin[i]
      v_odd[i]  = q[2i]*cos[i]   + q[2i-1]*sin[i]

    rope_table schema: (row_index, chunk_index, cos FLOAT[half], sin FLOAT[half])
    Output uses split even/odd layout: (row_index, chunk_index, v_even, v_odd)
    """
    half = str(chunk_size // 2)

    sql = (
        f"SELECT q.row_index, q.chunk_index, "
        f"list_transform(generate_series(1, {half}), "
        f"i -> CAST(q.v[2*i-1] * r.cos[i] - q.v[2*i] * r.sin[i] "
        f"AS FLOAT)) AS v_even, "
        f"list_transform(generate_series(1, {half}), "
        f"i -> CAST(q.v[2*i] * r.cos[i] + q.v[2*i-1] * r.sin[i] "
        f"AS FLOAT)) AS v_odd "
        f"FROM {q_table} q "
        f"JOIN {rope_table} r "
        f"ON r.chunk_index = q.chunk_index AND r.row_index = q.row_index"
    )

    return [(sql, out_table)]


# ---------------------------------------------------------------------------
# 5. Matrix multiplication — QK Attention  (Section 3.2.1)
# ---------------------------------------------------------------------------

def qk_attn_sql(q_rope_table: str, k_rope_table: str, out_table: str,
                num_q_heads: int, num_kv_heads: int,
                head_dim: int, chunk_size: int = 32) -> SqlSteps:
    """Query-Key attention scores with GQA.

    Paper §3.2.1: MatMul variant. Produces scalar (q_tok, k_tok, head_id, score).

    GQA join maps multiple Q heads to fewer KV heads.
    Causal mask: k.row_index <= q.row_index (Decision D4 — not in paper,
    required for correct autoregressive inference).

    Note: 1/sqrt(d_k) scaling is absorbed into W_Q during preprocessing
    (Section 3.1.2 constant folding).

    Inputs use RoPE split layout (v_even, v_odd columns).
    """
    chunks_per_head = head_dim // chunk_size
    group_size = num_q_heads // num_kv_heads

    head_stride = chunks_per_head * chunk_size  # raw-offset stride per head

    sql = (
        f"SELECT q.row_index AS q_tok, k.row_index AS k_tok, "
        f"q.chunk_index // {head_stride} AS head_id, "
        f"SUM(list_dot_product(q.v_even, k.v_even) + "
        f"list_dot_product(q.v_odd, k.v_odd)) AS score "
        f"FROM {q_rope_table} q "
        f"JOIN {k_rope_table} k "
        f"ON q.chunk_index % {head_stride} "
        f"= k.chunk_index % {head_stride} "
        f"AND q.chunk_index // {group_size * head_stride} "
        f"= k.chunk_index // {head_stride} "
        f"AND k.row_index <= q.row_index "
        f"GROUP BY q.row_index, k.row_index, "
        f"q.chunk_index // {head_stride}"
    )

    return [(sql, out_table)]


# ---------------------------------------------------------------------------
# 6. Normalisation — Softmax  (Section 3.2.1, Table 1)
# ---------------------------------------------------------------------------

def softmax_sql(input_table: str, out_table: str,
                *, stable: bool = False) -> SqlSteps:
    """Softmax normalisation over attention scores.

    Paper §3.2.1, Table 1:
      Normalize_{exp, SUM, div}
      WITH exp_sum AS (
        SELECT A.m, SUM(SUM(exp(A.chunk))) AS summation
        FROM A GROUP BY A.m)
      SELECT A.m, A.n, exp(A.chunk)/summation
      FROM A JOIN exp_sum ON A.m = exp_sum.m

    Input:  (q_tok, k_tok, head_id, score)
    Output: (q_tok, k_tok, head_id, attn_weight)

    The paper's formulation is a 2-step pattern. Decision D5: follow the paper
    exactly. When stable=True, use the numerically stable variant (subtract max)
    which is needed for large attention scores in production.
    """
    if stable:
        return _softmax_stable(input_table, out_table)

    # Paper's 2-step pattern (Decision D5)
    sum_t = out_table + "_expsum"

    step1 = (
        f"SELECT q_tok, head_id, SUM(exp(score)) AS summation "
        f"FROM {input_table} "
        f"GROUP BY q_tok, head_id"
    )

    step2 = (
        f"SELECT s.q_tok, s.k_tok, s.head_id, "
        f"CAST(exp(s.score) / e.summation AS FLOAT) AS attn_weight "
        f"FROM {input_table} s "
        f"JOIN {sum_t} e ON s.q_tok = e.q_tok AND s.head_id = e.head_id"
    )

    return [(step1, sum_t), (step2, out_table)]


def _softmax_stable(input_table: str, out_table: str) -> SqlSteps:
    """Numerically stable softmax (subtract max first).

    Not in the paper — provided as a configuration option for production use
    where large attention scores could cause exp() overflow.
    """
    max_t = out_table + "_max"
    exp_t = out_table + "_exp"
    sum_t = out_table + "_sum"

    step1 = (
        f"SELECT q_tok, head_id, MAX(score) AS max_score "
        f"FROM {input_table} "
        f"GROUP BY q_tok, head_id"
    )

    step2 = (
        f"SELECT s.q_tok, s.k_tok, s.head_id, "
        f"EXP(s.score - m.max_score) AS exp_val "
        f"FROM {input_table} s "
        f"JOIN {max_t} m ON s.q_tok = m.q_tok AND s.head_id = m.head_id"
    )

    step3 = (
        f"SELECT q_tok, head_id, SUM(exp_val) AS sum_exp "
        f"FROM {exp_t} "
        f"GROUP BY q_tok, head_id"
    )

    step4 = (
        f"SELECT e.q_tok, e.k_tok, e.head_id, "
        f"CAST(e.exp_val / s.sum_exp AS FLOAT) AS attn_weight "
        f"FROM {exp_t} e "
        f"JOIN {sum_t} s ON e.q_tok = s.q_tok AND e.head_id = s.head_id"
    )

    return [(step1, max_t), (step2, exp_t), (step3, sum_t), (step4, out_table)]


# ---------------------------------------------------------------------------
# 7. Matrix multiplication — Attention x V  (Section 3.2.1)
# ---------------------------------------------------------------------------

def attn_vmul_sql(attn_table: str, v_table: str, out_table: str,
                  num_q_heads: int, num_kv_heads: int,
                  head_dim: int, chunk_size: int = 32) -> SqlSteps:
    """Attention-weighted value sum.

    Paper §3.2.1: MatMul variant. Multiplies attention weights by V and
    re-chunks into standard chunked layout.

    attn_table: (q_tok, k_tok, head_id, attn_weight) — scalar layout
    v_table:    (row_index, chunk_index, v FLOAT[])   — standard chunked
    Output:     (row_index, chunk_index, v FLOAT[])   — standard chunked

    Steps: expand V to scalar → weighted sum → re-chunk.
    """
    chunks_per_head = head_dim // chunk_size
    group_size = num_q_heads // num_kv_heads

    cph = str(chunks_per_head)
    gs = str(group_size)
    cs = str(chunk_size)

    vs = out_table + "_vs"   # V scalar
    wt = out_table + "_w"    # weighted (scalar)

    # Step 1: expand V to scalar rows
    step1 = (
        f"SELECT row_index AS tok, chunk_index, "
        f"unnest(generate_series(0, {chunk_size - 1})) AS elem_pos, "
        f"CAST(unnest(v) AS FLOAT) AS val "
        f"FROM {v_table}"
    )

    # Step 2: weighted sum over k_tok
    # GQA: map Q head to KV head via integer division
    step2 = (
        f"SELECT s.q_tok, "
        f"s.head_id * {chunks_per_head * chunk_size} "
        f"+ v.chunk_index % {chunks_per_head * chunk_size} AS out_chunk_index, "
        f"v.elem_pos, "
        f"CAST(SUM(s.attn_weight * v.val) AS FLOAT) AS val "
        f"FROM {attn_table} s "
        f"JOIN {vs} v ON s.k_tok = v.tok "
        f"AND s.head_id // {gs} = v.chunk_index // {chunks_per_head * chunk_size} "
        f"GROUP BY s.q_tok, s.head_id, v.chunk_index, v.elem_pos"
    )

    # Step 3: re-chunk scalars into FLOAT[]
    step3 = (
        f"SELECT q_tok AS row_index, out_chunk_index AS chunk_index, "
        f"array_agg(val ORDER BY elem_pos) AS v "
        f"FROM {wt} "
        f"GROUP BY q_tok, out_chunk_index"
    )

    return [(step1, vs), (step2, wt), (step3, out_table)]


# ---------------------------------------------------------------------------
# 8. Element-wise function — SwiGLU  (Section 3.2.1)
# ---------------------------------------------------------------------------

def swiglu_sql(gate_table: str, up_table: str, out_table: str) -> SqlSteps:
    """SwiGLU activation: SiLU(gate) * up.

    Paper §3.2.1: Element-wise function category.
    SiLU(x) = x / (1 + exp(-x))  — sigmoid linear unit
    Output = SiLU(gate) * up, applied element-wise over chunks.

    Paper Table 1 reference: SELECT A.i, A.k, 1/(1+exp(-A.chunk)) FROM A
    (for sigmoid), extended to SiLU(gate) * up.
    """
    sql = (
        f"SELECT g.row_index, g.chunk_index, "
        f"list_transform(generate_series(1, len(g.v)), "
        f"i -> CAST((g.v[i] / (1.0 + exp(-g.v[i]))) * u.v[i] AS FLOAT)) AS v "
        f"FROM {gate_table} g "
        f"JOIN {up_table} u "
        f"ON g.row_index = u.row_index AND g.chunk_index = u.chunk_index"
    )
    return [(sql, out_table)]


# ---------------------------------------------------------------------------
# 9. Element-wise arithmetic — Residual Add  (Section 3.2.1)
# ---------------------------------------------------------------------------

def residual_add_sql(table_a: str, table_b: str, out_table: str) -> SqlSteps:
    """Element-wise residual addition.

    Paper §3.2.1, Table 1:
      SELECT A.i, A.k, A.chunk + B.chunk
      FROM A JOIN B ON A.i = B.i AND A.k = B.k
    """
    sql = (
        f"SELECT a.row_index, a.chunk_index, "
        f"list_transform(generate_series(1, len(a.v)), "
        f"i -> CAST(a.v[i] + b.v[i] AS FLOAT)) AS v "
        f"FROM {table_a} a "
        f"JOIN {table_b} b "
        f"ON a.row_index = b.row_index AND a.chunk_index = b.chunk_index"
    )
    return [(sql, out_table)]
