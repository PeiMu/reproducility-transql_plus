"""
Umbra (PostgreSQL) dialect SQL templates (paper §3.2 operators).

Mirrors ``transql_plus/sql_templates.py`` function-for-function.

Umbra is PostgreSQL wire-compatible but does NOT support all PostgreSQL
array features. Specifically, ``ARRAY(SELECT ...)`` subquery constructors
and multi-argument ``unnest(a, b)`` cause "numeric overflow" errors.

Strategy: use ``generate_series(1, N)`` cross-joined with array indexing
(``v[_i]``) for all element-wise operations, and ``array_agg(... ORDER BY
_i)`` to reassemble arrays. This uses only basic SQL features that Umbra
handles correctly.

Dialect translations:

    DuckDB                                 -> Umbra
    --------------------------------------    --------------------------------
    list_dot_product(a, b)                  SUM(a[_i] * b[_i]) via generate_series
    list_sum(list_transform(v, x->x*x))    SUM(v[_i] * v[_i]) via generate_series
    list_transform(v, x -> f(x))            array_agg(f(v[_i]) ORDER BY _i)
                                             via cross join with generate_series
    array_agg(val ORDER BY k)               array_agg(val ORDER BY k)  -- same
    a // b  (int floor div)                 a / b  (integer division in PG)
    CAST(x AS FLOAT)                        CAST(x AS REAL)

Chunk index convention (Decision D7) is unchanged — raw byte offsets.
"""

from __future__ import annotations

SqlStep = tuple[str, str]       # (sql_body, table_name)
SqlSteps = list[SqlStep]


# ---------------------------------------------------------------------------
# 1. Lookup table
# ---------------------------------------------------------------------------

def embed_lookup_sql(tokens_table: str, embed_table: str,
                     out_table: str) -> SqlSteps:
    """Embedding lookup via equi-join on token id (paper §3.2.2)."""
    sql = (
        f"SELECT t.pos AS row_index, "
        f"e.chunk_index AS chunk_index, e.v AS v "
        f"FROM {tokens_table} t "
        f"JOIN {embed_table} e ON t.token_id = e.row_index "
        f"ORDER BY t.pos, e.chunk_index"
    )
    return [(sql, out_table)]


# ---------------------------------------------------------------------------
# 2. Matrix multiplication
# ---------------------------------------------------------------------------

def matmul_sql(act_table: str, weight_table: str, out_table: str,
               chunk_size: int = 32) -> SqlSteps:
    """Chunked matrix multiplication (paper §3.2.1).

    Dot product via generate_series cross join: each (act, weight, chunk)
    row is expanded into chunk_size element-pairs, multiplied, and summed.
    """
    cs = str(chunk_size)
    dp = out_table + "_dp"

    step1 = (
        f"SELECT a.row_index AS act_row, w.row_index AS out_col, "
        f"SUM(a.v[_gs._i] * w.v[_gs._i]) AS val "
        f"FROM {act_table} a "
        f"JOIN {weight_table} w ON a.chunk_index = w.chunk_index "
        f"CROSS JOIN generate_series(1, {cs}) AS _gs(_i) "
        f"GROUP BY a.row_index, w.row_index"
    )

    step2 = (
        f"SELECT act_row AS row_index, "
        f"out_col - (out_col % {cs}) AS chunk_index, "
        f"array_agg(val ORDER BY out_col) AS v "
        f"FROM {dp} "
        f"GROUP BY act_row, out_col - (out_col % {cs})"
    )

    return [(step1, dp), (step2, out_table)]


# ---------------------------------------------------------------------------
# 3. RMSNorm
# ---------------------------------------------------------------------------

def rmsnorm_sql(input_table: str, gamma_table: str, out_table: str,
                hidden_dim: int, eps: float,
                chunk_size: int = 32) -> SqlSteps:
    """RMS normalisation with learnable gamma (paper §3.2.1).

    Step 1: sum of squares via generate_series cross join.
    Step 2: normalise+scale via generate_series + array_agg.
    """
    sq = out_table + "_sq"
    cs = str(chunk_size)
    eps_str = f"{eps:.10f}"

    step1 = (
        f"SELECT row_index, "
        f"SUM(v[_sq._i] * v[_sq._i]) AS sq_sum "
        f"FROM {input_table} "
        f"CROSS JOIN generate_series(1, {cs}) AS _sq(_i) "
        f"GROUP BY row_index"
    )

    step2 = (
        f"SELECT n.row_index AS row_index, n.chunk_index AS chunk_index, "
        f"array_agg(n.v[_gs._i] "
        f"/ sqrt(s.sq_sum / {hidden_dim}.0 + {eps_str}) "
        f"* w.v[_gs._i] ORDER BY _gs._i) AS v "
        f"FROM {input_table} n "
        f"JOIN {sq} s ON n.row_index = s.row_index "
        f"JOIN {gamma_table} w ON n.chunk_index = w.chunk_index "
        f"CROSS JOIN generate_series(1, {cs}) AS _gs(_i) "
        f"GROUP BY n.row_index, n.chunk_index"
    )

    return [(step1, sq), (step2, out_table)]


# ---------------------------------------------------------------------------
# 4. RoPE
# ---------------------------------------------------------------------------

def rope_sql(q_table: str, rope_table: str, out_table: str,
             chunk_size: int = 32) -> SqlSteps:
    """Rotary positional encoding (Decision D2).

    Uses generate_series(1, half) + array indexing + array_agg.
    """
    half = chunk_size // 2

    sql = (
        f"SELECT q.row_index AS row_index, q.chunk_index AS chunk_index, "
        f"array_agg(q.v[2*_gs._i-1] * r.cos[_gs._i] "
        f"- q.v[2*_gs._i] * r.sin[_gs._i] "
        f"ORDER BY _gs._i) AS v_even, "
        f"array_agg(q.v[2*_gs._i] * r.cos[_gs._i] "
        f"+ q.v[2*_gs._i-1] * r.sin[_gs._i] "
        f"ORDER BY _gs._i) AS v_odd "
        f"FROM {q_table} q "
        f"JOIN {rope_table} r "
        f"ON r.chunk_index = q.chunk_index AND r.row_index = q.row_index "
        f"CROSS JOIN generate_series(1, {half}) AS _gs(_i) "
        f"GROUP BY q.row_index, q.chunk_index"
    )

    return [(sql, out_table)]


# ---------------------------------------------------------------------------
# 5. QK attention
# ---------------------------------------------------------------------------

def qk_attn_sql(q_rope_table: str, k_rope_table: str, out_table: str,
                num_q_heads: int, num_kv_heads: int,
                head_dim: int, chunk_size: int = 32) -> SqlSteps:
    """Query-Key attention scores with GQA (paper §3.2.1).

    Dot product of v_even/v_odd (each half chunk_size long) via
    generate_series.
    """
    chunks_per_head = head_dim // chunk_size
    group_size = num_q_heads // num_kv_heads
    head_stride = chunks_per_head * chunk_size
    half = chunk_size // 2

    sql = (
        f"SELECT q.row_index AS q_tok, k.row_index AS k_tok, "
        f"q.chunk_index / {head_stride} AS head_id, "
        f"SUM(q.v_even[_gs._i] * k.v_even[_gs._i] "
        f"+ q.v_odd[_gs._i] * k.v_odd[_gs._i]) AS score "
        f"FROM {q_rope_table} q "
        f"JOIN {k_rope_table} k "
        f"ON q.chunk_index % {head_stride} "
        f"= k.chunk_index % {head_stride} "
        f"AND q.chunk_index / {group_size * head_stride} "
        f"= k.chunk_index / {head_stride} "
        f"AND k.row_index <= q.row_index "
        f"CROSS JOIN generate_series(1, {half}) AS _gs(_i) "
        f"GROUP BY q.row_index, k.row_index, "
        f"q.chunk_index / {head_stride}"
    )

    return [(sql, out_table)]


# ---------------------------------------------------------------------------
# 6. Softmax
# ---------------------------------------------------------------------------

def softmax_sql(input_table: str, out_table: str,
                *, stable: bool = False) -> SqlSteps:
    """Softmax normalisation over attention scores (paper §3.2.1)."""
    if stable:
        return _softmax_stable(input_table, out_table)

    sum_t = out_table + "_expsum"

    step1 = (
        f"SELECT q_tok, head_id, SUM(exp(score)) AS summation "
        f"FROM {input_table} "
        f"GROUP BY q_tok, head_id"
    )

    step2 = (
        f"SELECT s.q_tok AS q_tok, s.k_tok AS k_tok, s.head_id AS head_id, "
        f"exp(s.score) / e.summation AS attn_weight "
        f"FROM {input_table} s "
        f"JOIN {sum_t} e ON s.q_tok = e.q_tok AND s.head_id = e.head_id"
    )

    return [(step1, sum_t), (step2, out_table)]


def _softmax_stable(input_table: str, out_table: str) -> SqlSteps:
    max_t = out_table + "_max"
    exp_t = out_table + "_exp"
    sum_t = out_table + "_sum"

    step1 = (
        f"SELECT q_tok, head_id, MAX(score) AS max_score "
        f"FROM {input_table} "
        f"GROUP BY q_tok, head_id"
    )

    step2 = (
        f"SELECT s.q_tok AS q_tok, s.k_tok AS k_tok, s.head_id AS head_id, "
        f"exp(s.score - m.max_score) AS exp_val "
        f"FROM {input_table} s "
        f"JOIN {max_t} m ON s.q_tok = m.q_tok AND s.head_id = m.head_id"
    )

    step3 = (
        f"SELECT q_tok, head_id, SUM(exp_val) AS sum_exp "
        f"FROM {exp_t} "
        f"GROUP BY q_tok, head_id"
    )

    step4 = (
        f"SELECT e.q_tok AS q_tok, e.k_tok AS k_tok, e.head_id AS head_id, "
        f"e.exp_val / s.sum_exp AS attn_weight "
        f"FROM {exp_t} e "
        f"JOIN {sum_t} s ON e.q_tok = s.q_tok AND e.head_id = s.head_id"
    )

    return [(step1, max_t), (step2, exp_t), (step3, sum_t), (step4, out_table)]


# ---------------------------------------------------------------------------
# 7. Attention x V
# ---------------------------------------------------------------------------

def attn_vmul_sql(attn_table: str, v_table: str, out_table: str,
                  num_q_heads: int, num_kv_heads: int,
                  head_dim: int, chunk_size: int = 32) -> SqlSteps:
    """Attention-weighted value sum (paper §3.2.1).

    Step 1 expands V to scalar rows via generate_series + array indexing.
    """
    chunks_per_head = head_dim // chunk_size
    group_size = num_q_heads // num_kv_heads
    cph_cs = chunks_per_head * chunk_size

    vs = out_table + "_vs"
    wt = out_table + "_w"

    step1 = (
        f"SELECT row_index AS tok, chunk_index, "
        f"_gs.i AS elem_pos, "
        f"v[_gs.i + 1] AS val "
        f"FROM {v_table} "
        f"CROSS JOIN generate_series(0, {chunk_size - 1}) AS _gs(i)"
    )

    step2 = (
        f"SELECT s.q_tok AS q_tok, "
        f"s.head_id * {cph_cs} "
        f"+ v.chunk_index % {cph_cs} AS out_chunk_index, "
        f"v.elem_pos AS elem_pos, "
        f"SUM(s.attn_weight * v.val) AS val "
        f"FROM {attn_table} s "
        f"JOIN {vs} v ON s.k_tok = v.tok "
        f"AND s.head_id / {group_size} "
        f"= v.chunk_index / {cph_cs} "
        f"GROUP BY s.q_tok, s.head_id, v.chunk_index, v.elem_pos"
    )

    step3 = (
        f"SELECT q_tok AS row_index, out_chunk_index AS chunk_index, "
        f"array_agg(val ORDER BY elem_pos) AS v "
        f"FROM {wt} "
        f"GROUP BY q_tok, out_chunk_index"
    )

    return [(step1, vs), (step2, wt), (step3, out_table)]


# ---------------------------------------------------------------------------
# 8. SwiGLU
# ---------------------------------------------------------------------------

def swiglu_sql(gate_table: str, up_table: str, out_table: str,
               chunk_size: int = 32) -> SqlSteps:
    """SwiGLU activation: SiLU(gate) * up (paper §3.2.1).

    Uses generate_series + array indexing + array_agg.
    """
    cs = str(chunk_size)
    sql = (
        f"SELECT g.row_index AS row_index, g.chunk_index AS chunk_index, "
        f"array_agg((g.v[_gs._i] / (1.0 + exp(-g.v[_gs._i]))) "
        f"* u.v[_gs._i] ORDER BY _gs._i) AS v "
        f"FROM {gate_table} g "
        f"JOIN {up_table} u "
        f"ON g.row_index = u.row_index AND g.chunk_index = u.chunk_index "
        f"CROSS JOIN generate_series(1, {cs}) AS _gs(_i) "
        f"GROUP BY g.row_index, g.chunk_index"
    )
    return [(sql, out_table)]


# ---------------------------------------------------------------------------
# 9. Residual add
# ---------------------------------------------------------------------------

def residual_add_sql(table_a: str, table_b: str, out_table: str,
                     chunk_size: int = 32) -> SqlSteps:
    """Element-wise residual addition (paper §3.2.1).

    Uses generate_series + array indexing + array_agg.
    """
    cs = str(chunk_size)
    sql = (
        f"SELECT a.row_index AS row_index, a.chunk_index AS chunk_index, "
        f"array_agg(a.v[_gs._i] + b.v[_gs._i] "
        f"ORDER BY _gs._i) AS v "
        f"FROM {table_a} a "
        f"JOIN {table_b} b "
        f"ON a.row_index = b.row_index AND a.chunk_index = b.chunk_index "
        f"CROSS JOIN generate_series(1, {cs}) AS _gs(_i) "
        f"GROUP BY a.row_index, a.chunk_index"
    )
    return [(sql, out_table)]
