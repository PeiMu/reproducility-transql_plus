"""
ClickHouse-dialect SQL templates (paper §3.2 operators).

Mirrors ``transql_plus/sql_templates.py`` function-for-function. Every
template returns ``list[tuple[str, str]]`` — the same ``SqlSteps`` type —
so postopt_ch and runner_ch can consume them interchangeably with the
DuckDB pipeline.

Dialect translations (see reproduction_note.md D12 / probe JSON):

    DuckDB                                 -> ClickHouse
    --------------------------------------    --------------------------------
    list_dot_product(a, b)                  dotProduct(a, b)
    list_sum(arr)                           arraySum(arr)
    len(arr)                                length(arr)
    list_transform(v, x -> f(x))            arrayMap(x -> f(x), v)
    list_transform(generate_series(1, N),   arrayMap(i -> f(i), range(1, N+1))
                   i -> f(i))
    unnest(generate_series(0, N-1))         arrayJoin(range(0, N))
    unnest(v)                               arrayJoin(v)
    array_agg(val ORDER BY k)               arrayMap(x -> x.2, arraySort(
                                              x -> x.1, groupArray((k, val))))
    a // b   (int floor div)                intDiv(a, b)
    CAST(x AS FLOAT)                        CAST(x AS Float32)

Chunk index convention (Decision D7) is unchanged — raw byte offsets.
"""

from __future__ import annotations

SqlStep = tuple[str, str]       # (sql_body, table_name)
SqlSteps = list[SqlStep]


def _order_then_agg(order_key: str, val_expr: str) -> str:
    """Emit the ClickHouse idiom for ``array_agg(val ORDER BY k)``.

    Returns a single scalar expression usable as a ``SELECT`` projection:
    it groups ``(k, val)`` tuples, sorts the array by ``k``, and extracts
    the ``val`` field in order.
    """
    return (
        f"arrayMap(x -> x.2, arraySort(x -> x.1, "
        f"groupArray(({order_key}, {val_expr}))))"
    )


# ---------------------------------------------------------------------------
# 1. Lookup table
# ---------------------------------------------------------------------------

def embed_lookup_sql(tokens_table: str, embed_table: str,
                     out_table: str) -> SqlSteps:
    """Embedding lookup via equi-join on token id (paper §3.2.2).

    Every projected column carries an explicit ``AS`` alias: ClickHouse
    keeps qualified prefixes (``e.chunk_index``) as-is in the output
    schema of a multi-way join, which later breaks downstream queries
    that reference the table with a different alias (D12).
    """
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
    """Chunked matrix multiplication (paper §3.2.1)."""
    cs = str(chunk_size)
    dp = out_table + "_dp"

    step1 = (
        f"SELECT a.row_index AS act_row, w.row_index AS out_col, "
        f"SUM(dotProduct(a.v, w.v)) AS val "
        f"FROM {act_table} a "
        f"JOIN {weight_table} w ON a.chunk_index = w.chunk_index "
        f"GROUP BY a.row_index, w.row_index"
    )

    step2 = (
        f"SELECT act_row AS row_index, "
        f"out_col - (out_col % {cs}) AS chunk_index, "
        f"{_order_then_agg('out_col', 'val')} AS v "
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
    """RMS normalisation with learnable gamma (paper §3.2.1)."""
    sq = out_table + "_sq"
    eps_str = f"{eps:.10f}"

    step1 = (
        f"SELECT row_index, "
        f"SUM(arraySum(arrayMap(x -> x * x, v))) AS sq_sum "
        f"FROM {input_table} "
        f"GROUP BY row_index"
    )

    step2 = (
        f"SELECT n.row_index AS row_index, n.chunk_index AS chunk_index, "
        f"arrayMap(i -> CAST(n.v[i] / "
        f"sqrt(s.sq_sum / {hidden_dim}.0 + {eps_str}) "
        f"* w.v[i] AS Float32), range(1, length(n.v) + 1)) AS v "
        f"FROM {input_table} n "
        f"JOIN {sq} s ON n.row_index = s.row_index "
        f"JOIN {gamma_table} w ON n.chunk_index = w.chunk_index"
    )

    return [(step1, sq), (step2, out_table)]


# ---------------------------------------------------------------------------
# 4. RoPE
# ---------------------------------------------------------------------------

def rope_sql(q_table: str, rope_table: str, out_table: str,
             chunk_size: int = 32) -> SqlSteps:
    """Rotary positional encoding (Decision D2)."""
    half = chunk_size // 2
    half_s = str(half)

    sql = (
        f"SELECT q.row_index AS row_index, q.chunk_index AS chunk_index, "
        f"arrayMap(i -> CAST(q.v[2*i-1] * r.cos[i] - q.v[2*i] * r.sin[i] "
        f"AS Float32), range(1, {half_s} + 1)) AS v_even, "
        f"arrayMap(i -> CAST(q.v[2*i] * r.cos[i] + q.v[2*i-1] * r.sin[i] "
        f"AS Float32), range(1, {half_s} + 1)) AS v_odd "
        f"FROM {q_table} q "
        f"JOIN {rope_table} r "
        f"ON r.chunk_index = q.chunk_index AND r.row_index = q.row_index"
    )

    return [(sql, out_table)]


# ---------------------------------------------------------------------------
# 5. QK attention
# ---------------------------------------------------------------------------

def qk_attn_sql(q_rope_table: str, k_rope_table: str, out_table: str,
                num_q_heads: int, num_kv_heads: int,
                head_dim: int, chunk_size: int = 32) -> SqlSteps:
    """Query-Key attention scores with GQA (paper §3.2.1)."""
    chunks_per_head = head_dim // chunk_size
    group_size = num_q_heads // num_kv_heads
    head_stride = chunks_per_head * chunk_size

    sql = (
        f"SELECT q.row_index AS q_tok, k.row_index AS k_tok, "
        f"intDiv(q.chunk_index, {head_stride}) AS head_id, "
        f"SUM(dotProduct(q.v_even, k.v_even) + "
        f"dotProduct(q.v_odd, k.v_odd)) AS score "
        f"FROM {q_rope_table} q "
        f"JOIN {k_rope_table} k "
        f"ON q.chunk_index % {head_stride} "
        f"= k.chunk_index % {head_stride} "
        f"AND intDiv(q.chunk_index, {group_size * head_stride}) "
        f"= intDiv(k.chunk_index, {head_stride}) "
        f"AND k.row_index <= q.row_index "
        f"GROUP BY q.row_index, k.row_index, "
        f"intDiv(q.chunk_index, {head_stride})"
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
        f"CAST(exp(s.score) / e.summation AS Float32) AS attn_weight "
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
        f"CAST(e.exp_val / s.sum_exp AS Float32) AS attn_weight "
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
    """Attention-weighted value sum (paper §3.2.1)."""
    chunks_per_head = head_dim // chunk_size
    group_size = num_q_heads // num_kv_heads

    cph_cs = chunks_per_head * chunk_size
    gs = str(group_size)

    vs = out_table + "_vs"
    wt = out_table + "_w"

    # Step 1: expand V to scalar rows. ClickHouse requires the two
    # arrayJoins to run in lockstep; we achieve this by packaging them
    # through arrayEnumerate so the positions align.
    step1 = (
        f"SELECT row_index AS tok, chunk_index, "
        f"(arrayJoin(arrayZip(range(0, {chunk_size}), v)) "
        f") AS _elem, "
        f"_elem.1 AS elem_pos, "
        f"CAST(_elem.2 AS Float32) AS val "
        f"FROM {v_table}"
    )

    step2 = (
        f"SELECT s.q_tok AS q_tok, "
        f"s.head_id * {cph_cs} "
        f"+ v.chunk_index % {cph_cs} AS out_chunk_index, "
        f"v.elem_pos AS elem_pos, "
        f"CAST(SUM(s.attn_weight * v.val) AS Float32) AS val "
        f"FROM {attn_table} s "
        f"JOIN {vs} v ON s.k_tok = v.tok "
        f"AND intDiv(s.head_id, {gs}) "
        f"= intDiv(v.chunk_index, {cph_cs}) "
        f"GROUP BY s.q_tok, s.head_id, v.chunk_index, v.elem_pos"
    )

    step3 = (
        f"SELECT q_tok AS row_index, out_chunk_index AS chunk_index, "
        f"{_order_then_agg('elem_pos', 'val')} AS v "
        f"FROM {wt} "
        f"GROUP BY q_tok, out_chunk_index"
    )

    return [(step1, vs), (step2, wt), (step3, out_table)]


# ---------------------------------------------------------------------------
# 8. SwiGLU
# ---------------------------------------------------------------------------

def swiglu_sql(gate_table: str, up_table: str, out_table: str) -> SqlSteps:
    """SwiGLU activation: SiLU(gate) * up (paper §3.2.1)."""
    sql = (
        f"SELECT g.row_index AS row_index, g.chunk_index AS chunk_index, "
        f"arrayMap(i -> CAST((g.v[i] / (1.0 + exp(-g.v[i]))) * u.v[i] "
        f"AS Float32), range(1, length(g.v) + 1)) AS v "
        f"FROM {gate_table} g "
        f"JOIN {up_table} u "
        f"ON g.row_index = u.row_index AND g.chunk_index = u.chunk_index"
    )
    return [(sql, out_table)]


# ---------------------------------------------------------------------------
# 9. Residual add
# ---------------------------------------------------------------------------

def residual_add_sql(table_a: str, table_b: str, out_table: str) -> SqlSteps:
    """Element-wise residual addition (paper §3.2.1)."""
    sql = (
        f"SELECT a.row_index AS row_index, a.chunk_index AS chunk_index, "
        f"arrayMap(i -> CAST(a.v[i] + b.v[i] AS Float32), "
        f"range(1, length(a.v) + 1)) AS v "
        f"FROM {table_a} a "
        f"JOIN {table_b} b "
        f"ON a.row_index = b.row_index AND a.chunk_index = b.chunk_index"
    )
    return [(sql, out_table)]
