"""
Tensor computation DAG for the LLM forward pass.

Paper reference: Section 3.2 — "The forward-pass computation graph is exported
via ONNX and simplified using constant folding."

The DAG represents the full LLM forward pass as a directed acyclic graph of
tensor operations.  Nodes are stored in topological order.

is_shared flag: marks nodes whose output is consumed by multiple downstream
operators (§4.1: "critical nodes").  Used by CTE merging to decide which
intermediates must be materialised.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

from .config import ModelConfig


class TensorOpType(Enum):
    """Operator types covering the paper's five core categories (§3.2.1).

    All operators map to one or more of:
      1. Matrix multiplication
      2. Element-wise functions
      3. Element-wise arithmetic
      4. Shape manipulation
      5. Normalisation
    """
    EmbedLookup = auto()   # Lookup table (§3.2.2)
    RMSNorm     = auto()   # Normalisation: Normalize_{sq, SUM, rms_scale}
    MatMul      = auto()   # Matrix multiplication (§3.2.1, Table 1)
    RoPE        = auto()   # Elem-wise arith + shape manip (Decision D2)
    QKAttn      = auto()   # MatMul variant with GQA + causal mask (Decision D4)
    Softmax     = auto()   # Normalisation: Normalize_{exp, SUM, div}
    AttnVMul    = auto()   # MatMul variant: attention-weighted value sum
    SwiGLU      = auto()   # Element-wise function: SiLU(gate) * up
    ResidualAdd = auto()   # Element-wise arithmetic: A + B


@dataclass
class TensorDagNode:
    """A single operation in the computation graph.

    Attributes:
        id:           Unique node identifier.
        op_type:      Which operator to apply.
        output_table: Name of the temp table this op materialises.
        input_tables: Ordered list of input temp/weight table names.
        is_shared:    True when multiple downstream ops consume this output
                      (§4.1 critical node definition).
        params:       Op-specific parameters (e.g. hidden_dim, eps).
    """
    id: int
    op_type: TensorOpType
    output_table: str
    input_tables: list[str]
    is_shared: bool = False
    params: dict[str, str] = field(default_factory=dict)


class TensorComputeDAG:
    """Directed acyclic graph of the full LLM forward pass.

    Nodes are stored in topological order (build_llama3_8b guarantees this).
    """

    def __init__(self) -> None:
        self._nodes: list[TensorDagNode] = []
        self.output_node_id: int = -1

    @property
    def nodes(self) -> list[TensorDagNode]:
        return self._nodes

    def _add(self, op: TensorOpType, output: str,
             inputs: list[str], shared: bool,
             params: dict[str, str] | None = None) -> int:
        nid = len(self._nodes)
        self._nodes.append(TensorDagNode(
            id=nid, op_type=op, output_table=output,
            input_tables=inputs, is_shared=shared,
            params=params or {},
        ))
        return nid

    # ------------------------------------------------------------------
    # Factory: Llama3-8B
    # ------------------------------------------------------------------

    @classmethod
    def build_llama3_8b(cls, config: ModelConfig) -> TensorComputeDAG:
        """Construct the Llama3-8B forward-pass DAG.

        Matches the Llama3 architecture:
          embedding → 32 × (RMSNorm → Q/K/V → RoPE → QKAttn → Softmax →
          AttnVMul → O_proj → ResidualAdd → RMSNorm → gate/up → SwiGLU →
          down → ResidualAdd) → final RMSNorm → lm_head

        Source: AQP_middleware/transql/src/tensor_dag.cpp:46-249
        """
        dag = cls()
        cs = str(config.chunk_size)

        # -- Embedding lookup --
        dag._add(TensorOpType.EmbedLookup, "x_0",
                 ["input_tokens", "embed_tokens"],
                 shared=True)

        x_in = "x_0"

        for l in range(config.num_layers):
            pfx = f"l{l}_"

            def wt(name: str) -> str:
                return f"layer_{l}_{name}"

            # -- Pre-attention RMSNorm --
            norm1 = pfx + "norm1_out"
            dag._add(TensorOpType.RMSNorm, norm1,
                     [x_in, wt("norm1")],
                     shared=True,   # Q, K, V all read this
                     params={"hidden_dim": str(config.hidden_dim),
                             "eps": str(config.rms_norm_eps)})

            # -- Q / K / V projections --
            hd_s = str(config.hidden_dim)
            kv_s = str(config.kv_dim)

            q = pfx + "q"
            dag._add(TensorOpType.MatMul, q,
                     [norm1, wt("q_proj")], shared=False,
                     params={"chunk_size": cs, "contracted_dim": hd_s,
                             "out_dim": hd_s})

            k = pfx + "k"
            dag._add(TensorOpType.MatMul, k,
                     [norm1, wt("k_proj")], shared=False,
                     params={"chunk_size": cs, "contracted_dim": hd_s,
                             "out_dim": kv_s})

            v = pfx + "v"
            dag._add(TensorOpType.MatMul, v,
                     [norm1, wt("v_proj")], shared=True,  # used by AttnVMul
                     params={"chunk_size": cs, "contracted_dim": hd_s,
                             "out_dim": kv_s})

            # -- RoPE --
            q_rope = pfx + "q_rope"
            dag._add(TensorOpType.RoPE, q_rope,
                     [q, "rope"], shared=True,
                     params={"chunk_size": cs})

            k_rope = pfx + "k_rope"
            dag._add(TensorOpType.RoPE, k_rope,
                     [k, "rope"], shared=True,
                     params={"chunk_size": cs})

            # -- QK attention scores --
            qk = pfx + "qk_scores"
            dag._add(TensorOpType.QKAttn, qk,
                     [q_rope, k_rope], shared=False,
                     params={"num_q_heads": str(config.num_q_heads),
                             "num_kv_heads": str(config.num_kv_heads),
                             "head_dim": str(config.head_dim),
                             "chunk_size": cs})

            # -- Softmax --
            attn_w = pfx + "attn_weights"
            dag._add(TensorOpType.Softmax, attn_w,
                     [qk], shared=False)

            # -- Attention x V --
            attn_out = pfx + "attn_out"
            dag._add(TensorOpType.AttnVMul, attn_out,
                     [attn_w, v], shared=False,
                     params={"num_q_heads": str(config.num_q_heads),
                             "num_kv_heads": str(config.num_kv_heads),
                             "head_dim": str(config.head_dim),
                             "chunk_size": cs})

            # -- O projection --
            o = pfx + "o_proj"
            dag._add(TensorOpType.MatMul, o,
                     [attn_out, wt("o_proj")], shared=False,
                     params={"chunk_size": cs, "contracted_dim": hd_s,
                             "out_dim": hd_s})

            # -- Residual add 1 --
            x_attn = pfx + "x_after_attn"
            dag._add(TensorOpType.ResidualAdd, x_attn,
                     [x_in, o], shared=True)  # read by RMSNorm2 + ResidualAdd2

            # -- Pre-FFN RMSNorm --
            norm2 = pfx + "norm2_out"
            dag._add(TensorOpType.RMSNorm, norm2,
                     [x_attn, wt("norm2")], shared=True,
                     params={"hidden_dim": str(config.hidden_dim),
                             "eps": str(config.rms_norm_eps)})

            # -- Gate / Up --
            ffn_s = str(config.ffn_dim)

            gate = pfx + "gate"
            dag._add(TensorOpType.MatMul, gate,
                     [norm2, wt("gate_proj")], shared=False,
                     params={"chunk_size": cs, "contracted_dim": hd_s,
                             "out_dim": ffn_s})

            up = pfx + "up"
            dag._add(TensorOpType.MatMul, up,
                     [norm2, wt("up_proj")], shared=False,
                     params={"chunk_size": cs, "contracted_dim": hd_s,
                             "out_dim": ffn_s})

            # -- SwiGLU --
            ffn = pfx + "ffn_act"
            dag._add(TensorOpType.SwiGLU, ffn,
                     [gate, up], shared=False)

            # -- Down projection --
            down = pfx + "down"
            dag._add(TensorOpType.MatMul, down,
                     [ffn, wt("down_proj")], shared=False,
                     params={"chunk_size": cs, "contracted_dim": ffn_s,
                             "out_dim": hd_s})

            # -- Residual add 2 --
            x_out = pfx + "x_out"
            dag._add(TensorOpType.ResidualAdd, x_out,
                     [x_attn, down],
                     shared=(l < config.num_layers - 1))

            x_in = x_out

        # -- Final RMSNorm --
        dag._add(TensorOpType.RMSNorm, "final_norm_out",
                 [x_in, "final_norm"], shared=False,
                 params={"hidden_dim": str(config.hidden_dim),
                         "eps": str(config.rms_norm_eps)})

        # -- LM head --
        logits_id = dag._add(TensorOpType.MatMul, "logits",
                             ["final_norm_out", "lm_head"], shared=False,
                             params={"chunk_size": cs,
                                     "contracted_dim": str(config.hidden_dim),
                                     "out_dim": str(config.vocab_size)})

        dag.output_node_id = logits_id
        return dag

    # ------------------------------------------------------------------
    # Factory: JSON topology (ONNX export path)
    # ------------------------------------------------------------------

    _OP_TYPE_MAP: dict[str, TensorOpType] = {
        "EmbedLookup": TensorOpType.EmbedLookup,
        "RMSNorm":     TensorOpType.RMSNorm,
        "MatMul":      TensorOpType.MatMul,
        "RoPE":        TensorOpType.RoPE,
        "QKAttn":      TensorOpType.QKAttn,
        "Softmax":     TensorOpType.Softmax,
        "AttnVMul":    TensorOpType.AttnVMul,
        "SwiGLU":      TensorOpType.SwiGLU,
        "ResidualAdd": TensorOpType.ResidualAdd,
    }

    @classmethod
    def build_from_json(cls, json_path: str | Path) -> TensorComputeDAG:
        """Reconstruct DAG from a topology.json file.

        Paper §3.1: "The forward-pass computation graph is exported via ONNX."
        The ONNX extraction path produces a topology.json that this method
        reads, providing generality for arbitrary ONNX-exported models
        without hardcoding a DAG builder per architecture.

        JSON format:
            {"nodes": [{id, op_type, output_table, input_tables,
                        is_shared, params}, ...],
             "output_node_id": int}
        """
        with open(json_path) as f:
            data = json.load(f)

        dag = cls()
        for entry in data["nodes"]:
            op_str = entry["op_type"]
            if op_str not in cls._OP_TYPE_MAP:
                raise ValueError(
                    f"Unknown op_type '{op_str}' in topology.json. "
                    f"Known: {list(cls._OP_TYPE_MAP.keys())}")
            dag._add(
                cls._OP_TYPE_MAP[op_str],
                entry["output_table"],
                entry["input_tables"],
                entry.get("is_shared", False),
                entry.get("params", {}),
            )
        dag.output_node_id = data.get("output_node_id", len(dag._nodes) - 1)
        return dag
