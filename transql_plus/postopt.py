"""
Post-optimizations for TranSQL+ SQL generation.

Paper reference: Section 4.

Three optimisations, each independently toggleable:
  §4.1 — CTE merging (Algorithm 2: Temporary View Elimination)
  §4.2 — Table fusion (QKV and gate+up via UNION ALL with flag column)
  §4.3 — ROW2COL pivoting (CROSS JOIN + POSITIONAL JOIN MatMul)

Source: AQP_middleware/transql/src/transql_postopt.cpp
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .compute_dag import TensorComputeDAG, TensorDagNode, TensorOpType
from .dag_to_sql import expand_node
from .sql_templates import SqlStep, SqlSteps


# =====================================================================
# Options
# =====================================================================

@dataclass
class PostOptOptions:
    """Section 4 post-optimisation toggles."""
    cte_merge: bool = True       # §4.1
    table_fusion: bool = True    # §4.2
    row2col_pivot: bool = True   # §4.3
    pivot_width: int = 0         # §4.3: chunk-columns per pivot group (0 = all)
    subquery_width: int = 0      # §4.3: columns per CROSS JOIN CTE (0 = 1)
    # Heuristic skip thresholds (§4.3):
    # "skip ROW2COL for extremely wide pivots or very large chunks"
    max_pivot_chunks: int = 128  # skip if n_chunks > this
    max_chunk_size: int = 128    # skip if chunk_size > this


# =====================================================================
# Helpers
# =====================================================================

def _int(node: TensorDagNode, key: str) -> int:
    return int(node.params[key])


# =====================================================================
# §4.1  CTE Merging (Algorithm 2: Temporary View Elimination)
# =====================================================================

def _emit_cte_block(steps: list[SqlStep]) -> SqlStep:
    """Merge multiple (sql, name) pairs into a single WITH … AS block.

    Paper §4.1, Algorithm 2: consecutive non-critical steps are wrapped
    into a single WITH statement.  The last step becomes the body,
    materialised as a temp table.  All preceding steps become CTEs.

    If only one step, return it unchanged.
    """
    if len(steps) == 1:
        return steps[0]

    parts = []
    for sql, name in steps[:-1]:
        parts.append(f"{name} AS ({sql})")

    with_clause = "WITH " + ", ".join(parts)
    final_sql = with_clause + " " + steps[-1][0]
    return (final_sql, steps[-1][1])


# =====================================================================
# §4.2  Table Fusion
# =====================================================================

def fused_qkv_sql(norm_out: str,
                  q_weight: str, k_weight: str, v_weight: str,
                  qkv_out: str,
                  q_dim: int, kv_dim: int,
                  chunk_size: int) -> SqlSteps:
    """Fused QKV projection — single-table output (paper §4.2).

    Paper §4.2: "with a flag column distinguishing the projection type."

    Produces ONE table with (row_index, chunk_index, v, flag) containing
    all three projections.  Downstream ops filter by flag via cheap CTEs
    prepended by the CTE merge (see postopt_dag_to_sql).

    Steps:
      1. UNION ALL weight tables with flag column
      2. Single fused dot-product (one pass over the activation)
      3. Re-chunk all projections preserving flag column

    All dimensions from config, no hardcodes (Decision D6).
    """
    cs = chunk_size
    kv_off = q_dim
    v_off = q_dim + kv_dim

    w_name = qkv_out + "_w"
    dp_name = qkv_out + "_dp"

    # Step 1: fused weight table with flag column per paper §4.2
    w_sql = (
        f"SELECT row_index, chunk_index, v, 'Q' AS flag "
        f"FROM {q_weight} "
        f"UNION ALL "
        f"SELECT row_index, chunk_index, v, 'K' AS flag "
        f"FROM {k_weight} "
        f"UNION ALL "
        f"SELECT row_index, chunk_index, v, 'V' AS flag "
        f"FROM {v_weight}"
    )

    # Step 2: single fused dot-product
    dp_sql = (
        f"SELECT a.row_index AS act_row, "
        f"CASE w.flag WHEN 'Q' THEN w.row_index "
        f"WHEN 'K' THEN w.row_index + {kv_off} "
        f"WHEN 'V' THEN w.row_index + {v_off} END AS out_col, "
        f"w.flag, "
        f"SUM(list_dot_product(a.v, w.v)) AS val "
        f"FROM {norm_out} a "
        f"JOIN {w_name} w ON a.chunk_index = w.chunk_index "
        f"GROUP BY a.row_index, w.row_index, w.flag"
    )

    # Step 3: re-chunk all projections into (row_index, chunk_index, v, flag)
    # Each flag uses its own offset for local chunk_index (raw offset D7)
    rechunk_sql = (
        f"SELECT act_row AS row_index, "
        f"CASE flag "
        f"WHEN 'Q' THEN out_col - (out_col % {cs}) "
        f"WHEN 'K' THEN (out_col - {kv_off}) - ((out_col - {kv_off}) % {cs}) "
        f"WHEN 'V' THEN (out_col - {v_off}) - ((out_col - {v_off}) % {cs}) "
        f"END AS chunk_index, "
        f"array_agg(val ORDER BY out_col) AS v, "
        f"flag "
        f"FROM {dp_name} "
        f"GROUP BY act_row, flag, "
        f"CASE flag "
        f"WHEN 'Q' THEN out_col - (out_col % {cs}) "
        f"WHEN 'K' THEN (out_col - {kv_off}) - ((out_col - {kv_off}) % {cs}) "
        f"WHEN 'V' THEN (out_col - {v_off}) - ((out_col - {v_off}) % {cs}) "
        f"END"
    )

    return [
        (w_sql, w_name),
        (dp_sql, dp_name),
        (rechunk_sql, qkv_out),
    ]


def fused_gate_up_sql(norm_out: str,
                      gate_weight: str, up_weight: str,
                      gateup_out: str,
                      ffn_dim: int, chunk_size: int) -> SqlSteps:
    """Fused gate+up projection — single-table output (paper §4.2).

    Same pattern as QKV fusion but with two projections ('G'/'U').
    """
    cs = chunk_size

    w_name = gateup_out + "_w"
    dp_name = gateup_out + "_dp"

    w_sql = (
        f"SELECT row_index, chunk_index, v, 'G' AS flag "
        f"FROM {gate_weight} "
        f"UNION ALL "
        f"SELECT row_index, chunk_index, v, 'U' AS flag "
        f"FROM {up_weight}"
    )

    dp_sql = (
        f"SELECT a.row_index AS act_row, "
        f"CASE w.flag WHEN 'G' THEN w.row_index "
        f"WHEN 'U' THEN w.row_index + {ffn_dim} END AS out_col, "
        f"w.flag, "
        f"SUM(list_dot_product(a.v, w.v)) AS val "
        f"FROM {norm_out} a "
        f"JOIN {w_name} w ON a.chunk_index = w.chunk_index "
        f"GROUP BY a.row_index, w.row_index, w.flag"
    )

    rechunk_sql = (
        f"SELECT act_row AS row_index, "
        f"CASE flag "
        f"WHEN 'G' THEN out_col - (out_col % {cs}) "
        f"WHEN 'U' THEN (out_col - {ffn_dim}) - ((out_col - {ffn_dim}) % {cs}) "
        f"END AS chunk_index, "
        f"array_agg(val ORDER BY out_col) AS v, "
        f"flag "
        f"FROM {dp_name} "
        f"GROUP BY act_row, flag, "
        f"CASE flag "
        f"WHEN 'G' THEN out_col - (out_col % {cs}) "
        f"WHEN 'U' THEN (out_col - {ffn_dim}) - ((out_col - {ffn_dim}) % {cs}) "
        f"END"
    )

    return [
        (w_sql, w_name),
        (dp_sql, dp_name),
        (rechunk_sql, gateup_out),
    ]


# =====================================================================
# §4.3  ROW2COL Pivoting
# =====================================================================

def _chunk_offsets(chunk_start: int, chunk_count: int,
                   chunk_size: int) -> list[int]:
    """Return the raw-offset chunk_index values for a pivot group."""
    return [chunk_start + i * chunk_size for i in range(chunk_count)]


def pivot_sql(table_name: str, offsets: list[int]) -> str:
    """Pivot chunked rows into columns using DuckDB PIVOT.

    Paper §4.3: "The transformation uses PIVOT where available
    (e.g., SQL Server, DuckDB)."

    (row_index, chunk_index, v) → (row_index, "0", "32", "64", ...)

    Column names are the raw-offset chunk_index values (Decision D7).
    """
    in_list = ", ".join(str(o) for o in offsets)
    return (
        f"PIVOT {table_name} "
        f"ON chunk_index IN ({in_list}) "
        f"USING first(v) "
        f"GROUP BY row_index"
    )


def pivoted_matmul_dp(act_pivot: str, weight_pivot: str,
                      dp_out: str, offsets: list[int],
                      subquery_width: int) -> SqlSteps:
    """Generate pivoted MatMul dot-product CTEs.

    Paper §4.3 example:
      WITH c0 AS (SELECT A.row_id, B.row_id,
        DOT(A.chunk0, B.chunk0) AS v0
        FROM A_pivot CROSS JOIN B_pivot
        ORDER BY A.row_id, B.row_id),
      ... c63 AS (...)
      SELECT A.row_id AS i, B.row_id AS j,
        v0+...+v63 AS value_ij
      FROM c0 POSITIONAL JOIN c1 ...

    #projections per subquery = subquery_width (paper: "too many projections
    in one query can exceed parallelism").
    #subqueries = ceil(n_cols / subquery_width) (paper: "too many subqueries
    increase I/O").
    """
    if subquery_width <= 0:
        subquery_width = 1

    n_cols = len(offsets)
    steps: SqlSteps = []
    n_sq = (n_cols + subquery_width - 1) // subquery_width

    for sq in range(n_sq):
        ci = f"{dp_out}_sq{sq}"
        col_start = sq * subquery_width
        col_end = min(col_start + subquery_width, n_cols)

        dot_parts = []
        for c in range(col_start, col_end):
            # Column names from PIVOT are the raw offset values
            col = f'"{offsets[c]}"'
            dot_parts.append(f"list_dot_product(a.{col}, w.{col})")

        dot_expr = " + ".join(dot_parts)

        sql = (
            f"SELECT a.row_index AS act_row, w.row_index AS out_col, "
            f"{dot_expr} AS v{sq} "
            f"FROM {act_pivot} a CROSS JOIN {weight_pivot} w "
            f"ORDER BY a.row_index, w.row_index"
        )
        steps.append((sql, ci))

    # POSITIONAL JOIN reduction
    first = f"{dp_out}_sq0"
    sum_parts = [f"{dp_out}_sq{i}.v{i}" for i in range(n_sq)]
    sum_expr = " + ".join(sum_parts)

    from_parts = [first]
    for i in range(1, n_sq):
        from_parts.append(f"POSITIONAL JOIN {dp_out}_sq{i}")
    from_clause = " ".join(from_parts)

    final_sql = (
        f"SELECT {first}.act_row, {first}.out_col, "
        f"{sum_expr} AS val "
        f"FROM {from_clause}"
    )
    steps.append((final_sql, dp_out))
    return steps


def iter_pivot_specs(dag: TensorComputeDAG,
                     opts: PostOptOptions
                     ) -> list[tuple[str, str, list[int]]]:
    """Return (piv_name, source_weight_table, offsets) for every weight
    pivot that postopt_dag_to_sql would emit, deduplicated.

    Replicates the fusion + should_pivot decisions so the runner can
    pre-materialize exactly those weight pivots (Decision D9). Each tuple
    feeds directly into pivot_sql(source_weight_table, offsets); the
    resulting TEMP TABLE must be named piv_name so later CTE references
    in the generated SQL resolve to it.

    Pass the resulting set of piv_names as cached_weight_pivots= to
    postopt_dag_to_sql so it skips re-emitting the pivot step inside
    merged CTE blocks.
    """
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

        # Fused MatMul (Q/K/V or gate+up): fusion path uses UNION ALL,
        # not pivoting — individual weights are never pivoted.
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


def should_pivot(n_chunks: int, chunk_size: int,
                 opts: PostOptOptions) -> bool:
    """Heuristic: decide whether ROW2COL is worthwhile.

    Paper §4.3: "We skip ROW2COL for extremely wide pivots or very
    large chunks (CPU saturation)."  Only apply when:
      - chunk-parallel with moderate number of chunks
      - widened schema within engine limits
      - anticipated join-cardinality drop outweighs pivoting cost
    """
    return (n_chunks <= opts.max_pivot_chunks and
            chunk_size <= opts.max_chunk_size)


def pivoted_matmul_sql(act_table: str, weight_table: str,
                       out_table: str, n_chunks: int,
                       chunk_size: int, pivot_width: int = 0,
                       subquery_width: int = 0) -> SqlSteps:
    """Full pivoted MatMul pipeline.

    Paper §4.3: PIVOT → CROSS JOIN dot products → POSITIONAL JOIN → re-chunk.

    Trade-off (paper): "too many projections in one query can exceed
    parallelism, while too many subqueries increase I/O."

    Parameters from config (Decision D6):
      n_chunks:       number of chunks in the contracted dimension
      chunk_size:     elements per chunk
      pivot_width:    #projections — chunk-columns per pivoted sub-table
                      (0 = all at once)
      subquery_width: #projections per subquery — columns per CROSS JOIN CTE
                      (0 = 1 per CTE)
    """
    cs = chunk_size
    dp_name = out_table + "_dp"

    if pivot_width <= 0:
        pivot_width = n_chunks
    if subquery_width <= 0:
        subquery_width = 1

    n_groups = (n_chunks + pivot_width - 1) // pivot_width

    steps: SqlSteps = []
    group_dp_names: list[str] = []

    for g in range(n_groups):
        chunk_start_idx = g * pivot_width
        chunk_count = min(pivot_width, n_chunks - chunk_start_idx)
        chunk_start_offset = chunk_start_idx * chunk_size  # raw offset (D7)

        offsets = _chunk_offsets(chunk_start_offset, chunk_count, chunk_size)

        g_sfx = f"_g{g}" if n_groups > 1 else ""
        act_piv = f"{out_table}_act_piv{g_sfx}"
        wt_piv = f"{weight_table}_piv{g_sfx}"
        g_dp = f"{dp_name}_g{g}" if n_groups > 1 else dp_name

        steps.append((pivot_sql(act_table, offsets), act_piv))
        steps.append((pivot_sql(weight_table, offsets), wt_piv))

        dp_steps = pivoted_matmul_dp(act_piv, wt_piv, g_dp,
                                     offsets, subquery_width)
        steps.extend(dp_steps)
        group_dp_names.append(g_dp)

    # Sum across pivot groups via POSITIONAL JOIN
    if n_groups > 1:
        sum_parts = [f"{name}.val" for name in group_dp_names]
        sum_expr = " + ".join(sum_parts)

        from_parts = [group_dp_names[0]]
        for name in group_dp_names[1:]:
            from_parts.append(f"POSITIONAL JOIN {name}")
        from_clause = " ".join(from_parts)

        steps.append((
            f"SELECT {group_dp_names[0]}.act_row, "
            f"{group_dp_names[0]}.out_col, {sum_expr} AS val "
            f"FROM {from_clause}",
            dp_name,
        ))

    # Re-chunk (raw offset D7)
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
# Main entry point
# =====================================================================

def postopt_dag_to_sql(
    dag: TensorComputeDAG,
    opts: PostOptOptions | None = None,
    *,
    cached_weight_pivots: set[str] | None = None,
) -> SqlSteps:
    """Convert DAG to optimised SQL steps.

    Drop-in replacement for dag_to_sql() with Section 4 optimisations.

    Returns (sql_body, table_name) pairs.

    If cached_weight_pivots is provided, weight-pivot steps whose output
    name is in the set are suppressed — the CTE merge (§4.1) would
    otherwise re-emit them as CTEs inside a WITH block and shadow the
    pre-materialized TEMP TABLE of the same name. See Decision D9.
    """
    if opts is None:
        opts = PostOptOptions()

    nodes = dag.nodes
    output_id = dag.output_node_id

    # ── Phase 1: raw SQL generation with fusion ──

    fused_ids: set[int] = set()
    # Fused input mappings: original output → (fused_table, flag)
    # Downstream ops prepend filter CTEs to read from the fused table.
    fused_inputs: FusedInputMap = {}
    # Each group: (steps, is_shared, output_table)
    groups: list[tuple[SqlSteps, bool, str]] = []

    ni = 0
    while ni < len(nodes):
        if ni in fused_ids:
            ni += 1
            continue

        node = nodes[ni]
        at_output = (node.id == output_id)

        # ── Table fusion (§4.2) ──
        if opts.table_fusion and node.op_type == TensorOpType.MatMul:
            result = _try_fusion(nodes, ni, fused_ids)
            if result is not None:
                fused_steps, input_map, shared, skip = result
                groups.append((fused_steps, shared,
                               fused_steps[-1][1]))
                fused_inputs.update(input_map)
                fused_ids.update(skip)
                ni += 1
                continue

        # ── ROW2COL pivoting (§4.3) for non-fused MatMul ──
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
            # Falls through to default expansion if heuristic says skip

        # ── Default expansion ──
        steps = expand_node(node)

        # Prepend filter steps for inputs that come from fused tables.
        # Each filter is a cheap SELECT … WHERE flag = '?' that becomes
        # a CTE in the same WITH block as the consuming op's SQL.
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

    # ── Phase 2: CTE merge (§4.1, Algorithm 2) ──

    if not opts.cte_merge:
        flat: SqlSteps = []
        for steps, _, _ in groups:
            flat.extend(steps)
        return flat

    # Dependency-aware CTE merge (§4.1, Algorithm 2):
    #
    # A non-shared group can only be CTE-merged if its output_table is NOT
    # consumed by any group outside the current CTE chain.  Otherwise the
    # CTE name vanishes after the chain is flushed.
    #
    # Implementation:
    # 1. Map each group's output_table to its group index
    # 2. From the DAG, find which groups consume each output_table
    # 3. Assign "flush epochs" — which shared group flushes each group
    # 4. Force-materialise any non-shared group whose output crosses epochs

    # Step 1: map ALL step table names to their producing group index.
    # This is critical for fusion groups which produce multiple DAG-level
    # output tables (e.g., QKV fusion produces q, k, v).
    output_to_group: dict[str, int] = {}
    for gi, (steps, _, _) in enumerate(groups):
        for _, step_name in steps:
            output_to_group[step_name] = gi

    # Step 2: output_table → consuming group indices (from DAG)
    table_consumer_groups: dict[str, set[int]] = {}
    for node in nodes:
        for inp in node.input_tables:
            if inp in output_to_group:
                consumer_gi = output_to_group.get(node.output_table)
                if consumer_gi is not None:
                    table_consumer_groups.setdefault(inp, set()).add(
                        consumer_gi)

    # Step 2b: add fused table dependencies.
    # Filter steps reference the fused table, but the DAG doesn't know
    # about them.  Track which groups consume each fused table so the
    # CTE merge correctly detects cross-epoch dependencies.
    for orig_name, (fused_table, _flag) in fused_inputs.items():
        consumer_gi = output_to_group.get(orig_name)
        if consumer_gi is not None:
            table_consumer_groups.setdefault(fused_table, set()).add(
                consumer_gi)

    # Step 3: flush epoch — each group flushes with the next shared group
    flush_epoch: list[int] = [0] * len(groups)
    current_epoch = len(groups)  # sentinel for tail
    for gi in range(len(groups) - 1, -1, -1):
        if groups[gi][1]:  # is_shared
            current_epoch = gi
        flush_epoch[gi] = current_epoch

    # Step 4: find non-shared groups that must be force-materialised.
    # Check ALL step table names in each group, not just the group's
    # output_table — fusion groups have multiple externally-consumed tables.
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

    # Merge with forced materializations
    result: SqlSteps = []
    current_ctes: list[SqlStep] = []

    for gi, (steps, is_shared, _) in enumerate(groups):
        if gi in must_materialize:
            # Flush accumulated chain, then materialise this group's steps
            if current_ctes:
                result.append(_emit_cte_block(current_ctes))
                current_ctes.clear()
            # Find the first step with external consumers (consumed by
            # a group other than this one).  Steps before it are
            # internal-only and can be wrapped in a CTE block.
            # Steps from the boundary onward must be emitted individually
            # (e.g. fusion groups: q, k, v each consumed by later groups).
            boundary = len(steps)
            for si, (_, sname) in enumerate(steps):
                consumers = table_consumer_groups.get(sname, set())
                if consumers - {gi}:
                    boundary = si
                    break
            if boundary > 0:
                # Internal prefix → CTE block (materialises last prefix step)
                result.append(_emit_cte_block(steps[:boundary]))
            # Externally-consumed tail → individual steps
            result.extend(steps[boundary:])
        else:
            current_ctes.extend(steps)
            if is_shared:
                result.append(_emit_cte_block(current_ctes))
                current_ctes.clear()

    if current_ctes:
        result.append(_emit_cte_block(current_ctes))

    return result


# =====================================================================
# Fusion detection helpers
# =====================================================================

# FusedInputMap: original output table → (fused table name, flag value)
FusedInputMap = dict[str, tuple[str, str]]

# _try_fusion return: (steps, input_map, is_shared, skip_ids) or None
FusionResult = tuple[SqlSteps, FusedInputMap, bool, set[int]]


def _try_fusion(nodes: list[TensorDagNode], ni: int,
                fused_ids: set[int]) -> FusionResult | None:
    """Try to detect QKV or gate+up fusion pattern starting at node ni.

    Returns (fused_steps, fused_input_map, is_shared, skip_ids):
      - fused_steps: SQL steps producing the single fused table
      - fused_input_map: maps original output names to (fused_table, flag)
      - is_shared: True if fused table is consumed across CTE epochs
        (QKV: 3 consumers in different epochs; gate+up: 1 consumer)
      - skip_ids: DAG node indices absorbed by fusion
    """
    node = nodes[ni]

    # Need at least 2 more nodes for QKV (3 consecutive MatMuls)
    if ni + 2 < len(nodes):
        n1 = nodes[ni + 1]
        n2 = nodes[ni + 2]

        # QKV: 3 consecutive MatMuls with same activation input
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
                # QKV fused is consumed by q_rope, k_rope, attn_vmul
                # (3 groups in different CTE epochs) → must materialise
                return fused, input_map, True, {ni + 1, ni + 2}

    # Gate+up: 2 consecutive MatMuls with same activation, followed by SwiGLU
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
                # Gate+up consumed by SwiGLU only (1 group, same epoch)
                # → CTE-merges naturally, no need to materialise
                return fused, input_map, False, {ni + 1}

    return None


def _infer_qkv_dims(nodes: list[TensorDagNode], ni: int
                     ) -> tuple[int, int, int]:
    """Infer Q/KV dimensions from the nearest QKAttn node."""
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
    """Infer FFN dim by confirming SwiGLU pattern and reading down_proj params."""
    for j in range(ni + 2, len(nodes)):
        if (nodes[j].op_type == TensorOpType.SwiGLU and
                nodes[j].input_tables[0] == gate_node.output_table and
                nodes[j].input_tables[1] == up_node.output_table):
            # Found SwiGLU. Get FFN dim from the down_proj MatMul that follows.
            for k in range(j + 1, len(nodes)):
                if nodes[k].op_type == TensorOpType.MatMul:
                    # down_proj weight shape is (hidden_dim, ffn_dim).
                    # The contracted dimension is ffn_dim, which is also the
                    # output dimension of gate/up. We need this from the
                    # node's input — the SwiGLU output feeds into down_proj.
                    # Since we can't query the DB, derive from gate_proj:
                    # gate_proj is (ffn_dim, hidden_dim) → row_index up to ffn_dim.
                    # The simplest approach: read chunk_size from the MatMul node
                    # and compute ffn_dim from the down_proj's contracted chunks.
                    # But we need the actual dim. Store it in node params.
                    #
                    # Better: infer from the RMSNorm hidden_dim and the fact that
                    # gate_proj.output_dim = ffn_dim. But we can't access the DB.
                    #
                    # Since MatMul nodes now carry chunk_size, we can use it.
                    # For ffn_dim: look backwards for the nearest RMSNorm that
                    # has ffn_dim in its scope... but RMSNorm only has hidden_dim.
                    #
                    # Pragmatic: store ffn_dim in gate/up MatMul params.
                    cs = _int(gate_node, "chunk_size")
                    # ffn_dim must come from node params or config.
                    # Add "out_dim" to gate/up MatMul nodes during DAG construction.
                    if "out_dim" in gate_node.params:
                        return _int(gate_node, "out_dim"), cs
                    break
            break
    return -1, -1


def _infer_contracted_chunks(node: TensorDagNode,
                             nodes: list[TensorDagNode],
                             chunk_size: int) -> int:
    """Infer the number of chunks in the contracted dimension.

    For ROW2COL pivoting, we need the chunk count of the shared/contracted
    dimension (the one being dot-producted over). This equals
    contracted_dim / chunk_size.

    Derive from node params:
      - MatMul nodes now carry "contracted_dim" when set.
      - Fallback: look for nearby RMSNorm's hidden_dim (most MatMuls
        contract over hidden_dim).
    """
    if "contracted_dim" in node.params:
        return _int(node, "contracted_dim") // chunk_size

    # Heuristic: search nearby nodes for hidden_dim
    for n in nodes:
        if n.op_type == TensorOpType.RMSNorm and "hidden_dim" in n.params:
            return _int(n, "hidden_dim") // chunk_size

    raise ValueError(
        f"Cannot infer contracted dimension for node {node.output_table}. "
        f"Add 'contracted_dim' to node params."
    )
