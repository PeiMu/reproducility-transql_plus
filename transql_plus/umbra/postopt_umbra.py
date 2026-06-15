"""
Umbra (PostgreSQL) dialect post-optimisations (paper §4.1 / §4.2 / §4.3).

Structural mirror of ``transql_plus/postopt.py``. Differences live in the
SQL-emitting primitives:

    PIVOT (§4.3)              -> conditional aggregation with
                                 MAX(CASE WHEN chunk_index=N THEN v END)
    POSITIONAL JOIN (§4.3)    -> equi-JOIN on (act_row, out_col)
    list_dot_product           -> SUM(a[_i]*b[_i]) via generate_series
    a // b                     -> a / b  (integer division in PG)
    CAST AS FLOAT              -> CAST AS REAL

CTE merging (§4.1) is dialect-independent — standard WITH ... AS (...).
"""

from __future__ import annotations

from ..compute_dag import TensorComputeDAG, TensorDagNode, TensorOpType
from ..postopt import PostOptOptions, _chunk_offsets, should_pivot
from .sql_templates_umbra import (
    SqlStep,
    SqlSteps,
    embed_lookup_sql,
    matmul_sql,
    qk_attn_sql,
    residual_add_sql,
    rmsnorm_sql,
    rope_sql,
    softmax_sql,
    swiglu_sql,
    attn_vmul_sql,
)


# =====================================================================
# Per-node dispatch (mirror of dag_to_sql.expand_node, Umbra templates)
# =====================================================================

def _int(node: TensorDagNode, key: str) -> int:
    return int(node.params[key])


def _float(node: TensorDagNode, key: str) -> float:
    return float(node.params[key])


def expand_node_umbra(node: TensorDagNode) -> SqlSteps:
    inp = node.input_tables
    out = node.output_table

    match node.op_type:
        case TensorOpType.EmbedLookup:
            return embed_lookup_sql(inp[0], inp[1], out)
        case TensorOpType.MatMul:
            return matmul_sql(inp[0], inp[1], out,
                              _int(node, "chunk_size"))
        case TensorOpType.RMSNorm:
            return rmsnorm_sql(inp[0], inp[1], out,
                               _int(node, "hidden_dim"),
                               _float(node, "eps"))
        case TensorOpType.RoPE:
            return rope_sql(inp[0], inp[1], out,
                            _int(node, "chunk_size"))
        case TensorOpType.QKAttn:
            return qk_attn_sql(inp[0], inp[1], out,
                               _int(node, "num_q_heads"),
                               _int(node, "num_kv_heads"),
                               _int(node, "head_dim"),
                               _int(node, "chunk_size"))
        case TensorOpType.Softmax:
            return softmax_sql(inp[0], out, stable=True)
        case TensorOpType.AttnVMul:
            return attn_vmul_sql(inp[0], inp[1], out,
                                 _int(node, "num_q_heads"),
                                 _int(node, "num_kv_heads"),
                                 _int(node, "head_dim"),
                                 _int(node, "chunk_size"))
        case TensorOpType.SwiGLU:
            cs = int(node.params.get("chunk_size", 32))
            return swiglu_sql(inp[0], inp[1], out, cs)
        case TensorOpType.ResidualAdd:
            cs = int(node.params.get("chunk_size", 32))
            return residual_add_sql(inp[0], inp[1], out, cs)
        case _:
            raise ValueError(f"Unknown TensorOpType: {node.op_type}")


# =====================================================================
# §4.1 CTE Merging (standard SQL — same as DuckDB/ClickHouse)
# =====================================================================

def _emit_cte_block(steps: list[SqlStep]) -> SqlStep:
    if len(steps) == 1:
        return steps[0]

    parts = [f"{name} AS ({sql})" for sql, name in steps[:-1]]
    with_clause = "WITH " + ", ".join(parts)
    final_sql = with_clause + " " + steps[-1][0]
    return (final_sql, steps[-1][1])


# =====================================================================
# §4.2 Table fusion (Umbra dialect)
# =====================================================================

def fused_qkv_sql(norm_out: str,
                  q_weight: str, k_weight: str, v_weight: str,
                  qkv_out: str,
                  q_dim: int, kv_dim: int,
                  chunk_size: int) -> SqlSteps:
    """Fused QKV projection — Umbra dialect (paper §4.2)."""
    cs = chunk_size
    kv_off = q_dim
    v_off = q_dim + kv_dim

    w_name = qkv_out + "_w"
    dp_name = qkv_out + "_dp"

    w_sql = (
        f"SELECT row_index, chunk_index, v, 'Q' AS flag FROM {q_weight} "
        f"UNION ALL "
        f"SELECT row_index, chunk_index, v, 'K' AS flag FROM {k_weight} "
        f"UNION ALL "
        f"SELECT row_index, chunk_index, v, 'V' AS flag FROM {v_weight}"
    )

    dp_sql = (
        f"SELECT a.row_index AS act_row, "
        f"CASE w.flag WHEN 'Q' THEN w.row_index "
        f"WHEN 'K' THEN w.row_index + {kv_off} "
        f"WHEN 'V' THEN w.row_index + {v_off} END AS out_col, "
        f"w.flag AS flag, "
        f"SUM(a.v[_gs._i] * w.v[_gs._i]) AS val "
        f"FROM {norm_out} a "
        f"JOIN {w_name} w ON a.chunk_index = w.chunk_index "
        f"CROSS JOIN generate_series(1, {cs}) AS _gs(_i) "
        f"GROUP BY a.row_index, w.row_index, w.flag"
    )

    rechunk_sql = (
        f"SELECT act_row AS row_index, "
        f"CASE flag "
        f"WHEN 'Q' THEN out_col - (out_col % {cs}) "
        f"WHEN 'K' THEN (out_col - {kv_off}) - ((out_col - {kv_off}) % {cs}) "
        f"WHEN 'V' THEN (out_col - {v_off}) - ((out_col - {v_off}) % {cs}) "
        f"END AS chunk_index, "
        f"array_agg(val ORDER BY out_col) AS v, "
        f"flag AS flag "
        f"FROM {dp_name} "
        f"GROUP BY act_row, flag, "
        f"CASE flag "
        f"WHEN 'Q' THEN out_col - (out_col % {cs}) "
        f"WHEN 'K' THEN (out_col - {kv_off}) - ((out_col - {kv_off}) % {cs}) "
        f"WHEN 'V' THEN (out_col - {v_off}) - ((out_col - {v_off}) % {cs}) "
        f"END"
    )

    return [(w_sql, w_name), (dp_sql, dp_name), (rechunk_sql, qkv_out)]


def fused_gate_up_sql(norm_out: str,
                      gate_weight: str, up_weight: str,
                      gateup_out: str,
                      ffn_dim: int, chunk_size: int) -> SqlSteps:
    """Fused gate+up projection — Umbra dialect (paper §4.2)."""
    cs = chunk_size

    w_name = gateup_out + "_w"
    dp_name = gateup_out + "_dp"

    w_sql = (
        f"SELECT row_index, chunk_index, v, 'G' AS flag FROM {gate_weight} "
        f"UNION ALL "
        f"SELECT row_index, chunk_index, v, 'U' AS flag FROM {up_weight}"
    )

    dp_sql = (
        f"SELECT a.row_index AS act_row, "
        f"CASE w.flag WHEN 'G' THEN w.row_index "
        f"WHEN 'U' THEN w.row_index + {ffn_dim} END AS out_col, "
        f"w.flag AS flag, "
        f"SUM(a.v[_gs._i] * w.v[_gs._i]) AS val "
        f"FROM {norm_out} a "
        f"JOIN {w_name} w ON a.chunk_index = w.chunk_index "
        f"CROSS JOIN generate_series(1, {cs}) AS _gs(_i) "
        f"GROUP BY a.row_index, w.row_index, w.flag"
    )

    rechunk_sql = (
        f"SELECT act_row AS row_index, "
        f"CASE flag "
        f"WHEN 'G' THEN out_col - (out_col % {cs}) "
        f"WHEN 'U' THEN (out_col - {ffn_dim}) - ((out_col - {ffn_dim}) % {cs}) "
        f"END AS chunk_index, "
        f"array_agg(val ORDER BY out_col) AS v, "
        f"flag AS flag "
        f"FROM {dp_name} "
        f"GROUP BY act_row, flag, "
        f"CASE flag "
        f"WHEN 'G' THEN out_col - (out_col % {cs}) "
        f"WHEN 'U' THEN (out_col - {ffn_dim}) - ((out_col - {ffn_dim}) % {cs}) "
        f"END"
    )

    return [(w_sql, w_name), (dp_sql, dp_name), (rechunk_sql, gateup_out)]


# =====================================================================
# §4.3 ROW2COL pivoting — Umbra has no PIVOT and no POSITIONAL JOIN
# =====================================================================

def pivot_sql(table_name: str, offsets: list[int]) -> str:
    """Emit the DuckDB PIVOT equivalent as conditional aggregation.

    DuckDB:
        PIVOT tbl ON chunk_index IN (0, 32, ...) USING first(v)
              GROUP BY row_index

    Umbra (PostgreSQL):
        SELECT row_index,
               MAX(CASE WHEN chunk_index = 0 THEN v END) AS "0",
               MAX(CASE WHEN chunk_index = 32 THEN v END) AS "32",
               ...
        FROM tbl
        GROUP BY row_index
    """
    cols = ", ".join(
        f'MAX(CASE WHEN chunk_index = {o} THEN v END) AS "{o}"'
        for o in offsets
    )
    return (
        f"SELECT row_index, {cols} "
        f"FROM {table_name} "
        f"GROUP BY row_index"
    )


def pivoted_matmul_dp(act_pivot: str, weight_pivot: str,
                      dp_out: str, offsets: list[int],
                      chunk_size: int = 32) -> SqlSteps:
    """Pivoted MatMul dot-product — one query, no POSITIONAL JOIN.

    All chunk dot-products summed inline via generate_series across the
    CROSS JOIN of the two pivoted tables.
    """
    dot_parts = [
        f'SUM(a."{o}"[_gs._i] * w."{o}"[_gs._i])' for o in offsets
    ]
    dot_expr = " + ".join(dot_parts) if dot_parts else "0"

    sql = (
        f"SELECT a.row_index AS act_row, w.row_index AS out_col, "
        f"{dot_expr} AS val "
        f"FROM {act_pivot} a CROSS JOIN {weight_pivot} w "
        f"CROSS JOIN generate_series(1, {chunk_size}) AS _gs(_i) "
        f"GROUP BY a.row_index, w.row_index"
    )
    return [(sql, dp_out)]


def pivoted_matmul_sql(act_table: str, weight_table: str,
                       out_table: str, n_chunks: int,
                       chunk_size: int, pivot_width: int = 0,
                       subquery_width: int = 0) -> SqlSteps:
    """Full pivoted MatMul pipeline — Umbra dialect (paper §4.3).

    ``subquery_width`` is accepted for API parity but ignored: Umbra
    collapses the whole dot-product into one SELECT.
    """
    cs = chunk_size
    dp_name = out_table + "_dp"

    if pivot_width <= 0:
        pivot_width = n_chunks

    n_groups = (n_chunks + pivot_width - 1) // pivot_width
    steps: SqlSteps = []
    group_dp_names: list[str] = []

    for g in range(n_groups):
        chunk_start_idx = g * pivot_width
        chunk_count = min(pivot_width, n_chunks - chunk_start_idx)
        chunk_start_offset = chunk_start_idx * chunk_size

        offsets = _chunk_offsets(chunk_start_offset, chunk_count, chunk_size)

        g_sfx = f"_g{g}" if n_groups > 1 else ""
        act_piv = f"{out_table}_act_piv{g_sfx}"
        wt_piv = f"{weight_table}_piv{g_sfx}"
        g_dp = f"{dp_name}_g{g}" if n_groups > 1 else dp_name

        steps.append((pivot_sql(act_table, offsets), act_piv))
        steps.append((pivot_sql(weight_table, offsets), wt_piv))

        steps.extend(pivoted_matmul_dp(act_piv, wt_piv, g_dp, offsets,
                                       chunk_size))
        group_dp_names.append(g_dp)

    # Sum across pivot groups via equi-JOIN (no POSITIONAL JOIN in PG)
    if n_groups > 1:
        first = group_dp_names[0]
        from_parts = [f"{first}"]
        sum_parts = [f"{first}.val"]
        for name in group_dp_names[1:]:
            from_parts.append(
                f"JOIN {name} "
                f"ON {name}.act_row = {first}.act_row "
                f"AND {name}.out_col = {first}.out_col"
            )
            sum_parts.append(f"{name}.val")
        sum_expr = " + ".join(sum_parts)

        steps.append((
            f"SELECT {first}.act_row AS act_row, "
            f"{first}.out_col AS out_col, "
            f"{sum_expr} AS val "
            f"FROM {' '.join(from_parts)}",
            dp_name,
        ))

    rechunk = (
        f"SELECT act_row AS row_index, "
        f"out_col - (out_col % {cs}) AS chunk_index, "
        f"array_agg(val ORDER BY out_col) AS v "
        f"FROM {dp_name} "
        f"GROUP BY act_row, out_col - (out_col % {cs})"
    )
    steps.append((rechunk, out_table))
    return steps


# =====================================================================
# Fusion detection — re-used from postopt.py with local SQL emitters
# =====================================================================

FusedInputMap = dict[str, tuple[str, str]]
FusionResult = tuple[SqlSteps, FusedInputMap, bool, set[int]]


def _try_fusion(nodes: list[TensorDagNode], ni: int,
                fused_ids: set[int]) -> FusionResult | None:
    """QKV / gate+up fusion detector (same logic as postopt.py)."""
    node = nodes[ni]

    if ni + 2 < len(nodes):
        n1 = nodes[ni + 1]
        n2 = nodes[ni + 2]

        if (n1.op_type == TensorOpType.MatMul and
                n2.op_type == TensorOpType.MatMul and
                ni + 1 not in fused_ids and ni + 2 not in fused_ids and
                node.input_tables[0] == n1.input_tables[0] and
                node.input_tables[0] == n2.input_tables[0]):

            q_dim, kv_dim, cs = _infer_qkv_dims(nodes, ni)
            if q_dim > 0:
                qkv_out = node.output_table + "_qkv"
                fused = fused_qkv_sql(
                    node.input_tables[0],
                    node.input_tables[1], n1.input_tables[1],
                    n2.input_tables[1],
                    qkv_out, q_dim, kv_dim, cs,
                )
                input_map: FusedInputMap = {
                    node.output_table: (qkv_out, 'Q'),
                    n1.output_table:   (qkv_out, 'K'),
                    n2.output_table:   (qkv_out, 'V'),
                }
                return fused, input_map, True, {ni + 1, ni + 2}

    if ni + 1 < len(nodes):
        n1 = nodes[ni + 1]
        if (n1.op_type == TensorOpType.MatMul and
                ni + 1 not in fused_ids and
                node.input_tables[0] == n1.input_tables[0]):

            ffn_dim, cs = _infer_gate_up_dims(nodes, ni, node, n1)
            if ffn_dim > 0:
                gateup_out = node.output_table + "_gateup"
                fused = fused_gate_up_sql(
                    node.input_tables[0],
                    node.input_tables[1], n1.input_tables[1],
                    gateup_out, ffn_dim, cs,
                )
                input_map = {
                    node.output_table: (gateup_out, 'G'),
                    n1.output_table:   (gateup_out, 'U'),
                }
                return fused, input_map, False, {ni + 1}

    return None


def _infer_qkv_dims(nodes: list[TensorDagNode], ni: int
                     ) -> tuple[int, int, int]:
    for j in range(ni + 3, len(nodes)):
        if nodes[j].op_type == TensorOpType.QKAttn:
            num_q = _int(nodes[j], "num_q_heads")
            num_kv = _int(nodes[j], "num_kv_heads")
            hd = _int(nodes[j], "head_dim")
            cs = _int(nodes[j], "chunk_size")
            return num_q * hd, num_kv * hd, cs
    return -1, -1, -1


def _infer_gate_up_dims(nodes: list[TensorDagNode], ni: int,
                        gate_node: TensorDagNode,
                        up_node: TensorDagNode) -> tuple[int, int]:
    for j in range(ni + 2, len(nodes)):
        if (nodes[j].op_type == TensorOpType.SwiGLU and
                nodes[j].input_tables[0] == gate_node.output_table and
                nodes[j].input_tables[1] == up_node.output_table):
            for k in range(j + 1, len(nodes)):
                if nodes[k].op_type == TensorOpType.MatMul:
                    cs = _int(gate_node, "chunk_size")
                    if "out_dim" in gate_node.params:
                        return _int(gate_node, "out_dim"), cs
                    break
            break
    return -1, -1


def _infer_contracted_chunks(node: TensorDagNode,
                             nodes: list[TensorDagNode],
                             chunk_size: int) -> int:
    if "contracted_dim" in node.params:
        return _int(node, "contracted_dim") // chunk_size
    for n in nodes:
        if n.op_type == TensorOpType.RMSNorm and "hidden_dim" in n.params:
            return _int(n, "hidden_dim") // chunk_size
    raise ValueError(
        f"Cannot infer contracted dimension for node {node.output_table}."
    )


# =====================================================================
# iter_pivot_specs — for runner pre-materialization (Decision D9)
# =====================================================================

def iter_pivot_specs(dag: TensorComputeDAG,
                     opts: PostOptOptions
                     ) -> list[tuple[str, str, list[int]]]:
    """Return weight-pivot specs the runner should pre-materialize."""
    if not opts.row2col_pivot:
        return []

    nodes = dag.nodes
    fused_ids: set[int] = set()
    specs: list[tuple[str, str, list[int]]] = []
    seen: set[str] = set()

    ni = 0
    while ni < len(nodes):
        if ni in fused_ids:
            ni += 1
            continue

        node = nodes[ni]

        if opts.table_fusion and node.op_type == TensorOpType.MatMul:
            result = _try_fusion(nodes, ni, fused_ids)
            if result is not None:
                _, _, _, skip = result
                fused_ids.update(skip)
                ni += 1
                continue

        if node.op_type == TensorOpType.MatMul:
            cs = _int(node, "chunk_size")
            n_chunks = _infer_contracted_chunks(node, nodes, cs)

            if should_pivot(n_chunks, cs, opts):
                pivot_width = (opts.pivot_width
                               if opts.pivot_width > 0 else n_chunks)
                n_groups = (n_chunks + pivot_width - 1) // pivot_width
                weight_table = node.input_tables[1]

                for g in range(n_groups):
                    chunk_start_idx = g * pivot_width
                    chunk_count = min(pivot_width,
                                      n_chunks - chunk_start_idx)
                    chunk_start_offset = chunk_start_idx * cs
                    offsets = _chunk_offsets(chunk_start_offset,
                                             chunk_count, cs)

                    g_sfx = f"_g{g}" if n_groups > 1 else ""
                    piv_name = f"{weight_table}_piv{g_sfx}"

                    if piv_name not in seen:
                        seen.add(piv_name)
                        specs.append((piv_name, weight_table, offsets))

        ni += 1

    return specs


# =====================================================================
# Main entry point (mirror of postopt.postopt_dag_to_sql)
# =====================================================================

def postopt_dag_to_sql_umbra(
    dag: TensorComputeDAG,
    opts: PostOptOptions | None = None,
    *,
    cached_weight_pivots: set[str] | None = None,
) -> SqlSteps:
    """Convert DAG to optimised Umbra (PostgreSQL) SQL steps."""
    if opts is None:
        opts = PostOptOptions()

    nodes = dag.nodes
    output_id = dag.output_node_id

    fused_ids: set[int] = set()
    fused_inputs: FusedInputMap = {}
    groups: list[tuple[SqlSteps, bool, str]] = []

    ni = 0
    while ni < len(nodes):
        if ni in fused_ids:
            ni += 1
            continue

        node = nodes[ni]
        at_output = (node.id == output_id)

        if opts.table_fusion and node.op_type == TensorOpType.MatMul:
            result = _try_fusion(nodes, ni, fused_ids)
            if result is not None:
                fused_steps, input_map, shared, skip = result
                groups.append((fused_steps, shared, fused_steps[-1][1]))
                fused_inputs.update(input_map)
                fused_ids.update(skip)
                ni += 1
                continue

        if opts.row2col_pivot and node.op_type == TensorOpType.MatMul:
            cs = _int(node, "chunk_size")
            n_chunks = _infer_contracted_chunks(node, nodes, cs)

            if should_pivot(n_chunks, cs, opts):
                pivoted = pivoted_matmul_sql(
                    node.input_tables[0], node.input_tables[1],
                    node.output_table, n_chunks, cs,
                    opts.pivot_width, opts.subquery_width,
                )
                if cached_weight_pivots:
                    pivoted = [(sql, name) for sql, name in pivoted
                               if name not in cached_weight_pivots]
                groups.append((pivoted, node.is_shared or at_output,
                               node.output_table))
                ni += 1
                continue

        steps = expand_node_umbra(node)

        filter_steps: SqlSteps = []
        for inp in node.input_tables:
            if inp in fused_inputs:
                fused_table, flag = fused_inputs[inp]
                filter_steps.append((
                    f"SELECT row_index, chunk_index, v "
                    f"FROM {fused_table} WHERE flag = '{flag}'",
                    inp,
                ))
        if filter_steps:
            steps = filter_steps + steps

        groups.append((steps, node.is_shared or at_output,
                       node.output_table))
        ni += 1

    # ── Phase 2: CTE merge (dependency-aware, verbatim from postopt.py) ──
    if not opts.cte_merge:
        flat: SqlSteps = []
        for steps, _, _ in groups:
            flat.extend(steps)
        return flat

    output_to_group: dict[str, int] = {}
    for gi, (steps, _, _) in enumerate(groups):
        for _, step_name in steps:
            output_to_group[step_name] = gi

    table_consumer_groups: dict[str, set[int]] = {}
    for node in nodes:
        for inp in node.input_tables:
            if inp in output_to_group:
                consumer_gi = output_to_group.get(node.output_table)
                if consumer_gi is not None:
                    table_consumer_groups.setdefault(inp, set()).add(
                        consumer_gi)

    for orig_name, (fused_table, _flag) in fused_inputs.items():
        consumer_gi = output_to_group.get(orig_name)
        if consumer_gi is not None:
            table_consumer_groups.setdefault(fused_table, set()).add(
                consumer_gi)

    flush_epoch: list[int] = [0] * len(groups)
    current_epoch = len(groups)
    for gi in range(len(groups) - 1, -1, -1):
        if groups[gi][1]:
            current_epoch = gi
        flush_epoch[gi] = current_epoch

    must_materialize: set[int] = set()
    for gi, (steps, is_shared, _) in enumerate(groups):
        if is_shared:
            continue
        my_epoch = flush_epoch[gi]
        for _, step_name in steps:
            for cgi in table_consumer_groups.get(step_name, set()):
                if flush_epoch[cgi] != my_epoch:
                    must_materialize.add(gi)
                    break
            if gi in must_materialize:
                break

    result: SqlSteps = []
    current_ctes: list[SqlStep] = []

    for gi, (steps, is_shared, _) in enumerate(groups):
        if gi in must_materialize:
            if current_ctes:
                result.append(_emit_cte_block(current_ctes))
                current_ctes.clear()
            boundary = len(steps)
            for si, (_, sname) in enumerate(steps):
                consumers = table_consumer_groups.get(sname, set())
                if consumers - {gi}:
                    boundary = si
                    break
            if boundary > 0:
                result.append(_emit_cte_block(steps[:boundary]))
            result.extend(steps[boundary:])
        else:
            current_ctes.extend(steps)
            if is_shared:
                result.append(_emit_cte_block(current_ctes))
                current_ctes.clear()

    if current_ctes:
        result.append(_emit_cte_block(current_ctes))

    return result


def dag_to_sql_umbra(dag: TensorComputeDAG) -> SqlSteps:
    """Baseline DAG-to-SQL expansion (no post-opts)."""
    steps: SqlSteps = []
    for node in dag.nodes:
        steps.extend(expand_node_umbra(node))
    return steps
