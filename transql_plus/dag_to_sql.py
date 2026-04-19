"""
Baseline DAG-to-SQL expansion (no post-optimisation).

Paper reference: Section 3.2.3 — "SQL code generation".

Iterates the DAG in topological order and expands each node into one or more
SQL steps using the templates from sql_templates.py.

Source: AQP_middleware/transql/src/dag_to_tree.cpp:30-122
"""

from __future__ import annotations

from .compute_dag import TensorComputeDAG, TensorDagNode, TensorOpType
from .sql_templates import (
    SqlSteps,
    embed_lookup_sql,
    matmul_sql,
    rmsnorm_sql,
    rope_sql,
    qk_attn_sql,
    softmax_sql,
    attn_vmul_sql,
    swiglu_sql,
    residual_add_sql,
)


def _int(node: TensorDagNode, key: str) -> int:
    return int(node.params[key])


def _float(node: TensorDagNode, key: str) -> float:
    return float(node.params[key])


def expand_node(node: TensorDagNode) -> SqlSteps:
    """Expand a single DAG node into raw SQL steps."""
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
            return softmax_sql(inp[0], out)
        case TensorOpType.AttnVMul:
            return attn_vmul_sql(inp[0], inp[1], out,
                                 _int(node, "num_q_heads"),
                                 _int(node, "num_kv_heads"),
                                 _int(node, "head_dim"),
                                 _int(node, "chunk_size"))
        case TensorOpType.SwiGLU:
            return swiglu_sql(inp[0], inp[1], out)
        case TensorOpType.ResidualAdd:
            return residual_add_sql(inp[0], inp[1], out)
        case _:
            raise ValueError(f"Unknown TensorOpType: {node.op_type}")


def dag_to_sql(dag: TensorComputeDAG) -> SqlSteps:
    """Convert DAG to a flat list of SQL steps (no optimisation).

    Each step is (sql_body, table_name).  The runner wraps each as:
        CREATE TEMP TABLE table_name AS (sql_body)
    """
    steps: SqlSteps = []
    for node in dag.nodes:
        steps.extend(expand_node(node))
    return steps
