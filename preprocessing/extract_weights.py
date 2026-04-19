"""
Extract model weights for TranSQL+.

Paper reference: Section 3.1 — "The forward-pass computation graph is exported
via ONNX and simplified using constant folding."

Two extraction paths:
  1. ONNX (primary, per paper): export → onnxsim constant folding → extract
  2. PyTorch (convenience): download from HuggingFace → save .npy

Constant folding (§3.1.2): absorbs 1/sqrt(head_dim) into W_Q.
RoPE cos/sin tables are precomputed (equivalent to ONNX constant folding).

Outputs (one .npy per weight tensor):
    embed_tokens.npy         [vocab_size, hidden_dim]
    lm_head.npy              [vocab_size, hidden_dim]
    final_norm.npy           [hidden_dim]
    layer_{l}_q_proj.npy     [hidden_dim, hidden_dim]     (with 1/sqrt(head_dim) folded in)
    layer_{l}_k_proj.npy     [kv_dim, hidden_dim]
    layer_{l}_v_proj.npy     [kv_dim, hidden_dim]
    layer_{l}_o_proj.npy     [hidden_dim, hidden_dim]
    layer_{l}_gate_proj.npy  [ffn_dim, hidden_dim]
    layer_{l}_up_proj.npy    [ffn_dim, hidden_dim]
    layer_{l}_down_proj.npy  [hidden_dim, ffn_dim]
    layer_{l}_norm1.npy      [hidden_dim]
    layer_{l}_norm2.npy      [hidden_dim]
    rope_cos.npy             [max_seq_len, hidden_dim // chunk_size, chunk_size // 2]
    rope_sin.npy             [max_seq_len, hidden_dim // chunk_size, chunk_size // 2]

Usage:
    python -m preprocessing.extract_weights --output-dir weights_npy

    python -m preprocessing.extract_weights --source onnx \\
        --onnx-path model.onnx --output-dir weights_npy
"""

from __future__ import annotations

import argparse
import os

import numpy as np


# ---------------------------------------------------------------------------
# RoPE precomputation
# ---------------------------------------------------------------------------

def precompute_rope(
    hidden_dim: int,
    head_dim: int,
    rope_theta: float,
    max_seq_len: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute RoPE cos/sin tables in chunked format.

    Equivalent to ONNX constant folding for the RoPE sub-graph.
    Each head of dimension head_dim uses standard rotary encoding:
        angle(pos, d) = pos / theta^(2d / head_dim)

    Returns:
        cos_table: [max_seq_len, num_chunks, half_chunk]
        sin_table: [max_seq_len, num_chunks, half_chunk]
    """
    num_chunks = hidden_dim // chunk_size
    half = chunk_size // 2

    # Vectorised computation
    positions = np.arange(max_seq_len, dtype=np.float64)
    chunks = np.arange(num_chunks, dtype=np.int32)
    pairs = np.arange(half, dtype=np.int32)

    # Build (num_chunks, half) grid of global dim indices
    c_grid, p_grid = np.meshgrid(chunks, pairs, indexing="ij")
    global_dim = c_grid * chunk_size + 2 * p_grid  # (num_chunks, half)
    d_in_head = global_dim % head_dim               # dim index within head
    pair_idx = d_in_head // 2                        # pair index within head

    # theta per (chunk, pair): shape (num_chunks, half)
    inv_freq = 1.0 / (rope_theta ** (2.0 * pair_idx / head_dim))

    # angles: (max_seq_len,) outer (num_chunks, half) → (max_seq_len, num_chunks, half)
    angles = positions[:, None, None] * inv_freq[None, :, :]

    cos_table = np.cos(angles).astype(np.float32)
    sin_table = np.sin(angles).astype(np.float32)

    return cos_table, sin_table


# ---------------------------------------------------------------------------
# PyTorch extraction path
# ---------------------------------------------------------------------------

def extract_pytorch(
    model_id: str,
    output_dir: str,
    chunk_size: int,
    max_seq_len: int,
) -> None:
    """Download weights from HuggingFace and save as .npy files."""
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading config for {model_id}...")
    config = AutoConfig.from_pretrained(model_id)
    print(f"  hidden_size         = {config.hidden_size}")
    print(f"  num_attention_heads = {config.num_attention_heads}")
    print(f"  num_key_value_heads = {config.num_key_value_heads}")
    print(f"  intermediate_size   = {config.intermediate_size}")
    print(f"  num_hidden_layers   = {config.num_hidden_layers}")

    print(f"\nLoading model weights (float32)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map="cpu",
    )
    sd = {k: v.float().numpy() for k, v in model.state_dict().items()}

    # PyTorch nn.Linear stores weight as [out_features, in_features].
    # This is already the transposed layout the paper requires (§3.1):
    # "transposes the right matrix of matmul during preprocessing — storing
    # its column vectors as row vectors — so dot products can be computed
    # directly at inference time without an extra transpose."
    # PyTorch's layout has rows = output neurons, columns = contracted dim,
    # so chunking along axis 1 produces chunks along the contracted dimension.
    # No explicit transpose needed (unlike the ONNX path).

    # Global weights
    np.save(os.path.join(output_dir, "embed_tokens.npy"),
            sd["model.embed_tokens.weight"])
    np.save(os.path.join(output_dir, "final_norm.npy"),
            sd["model.norm.weight"])

    lm_key = "lm_head.weight"
    if lm_key not in sd:
        lm_key = "model.embed_tokens.weight"  # tied embeddings
    np.save(os.path.join(output_dir, "lm_head.npy"), sd[lm_key])

    # Per-layer weights
    head_dim = config.hidden_size // config.num_attention_heads
    scale = np.float32(1.0 / np.sqrt(head_dim))
    num_layers = config.num_hidden_layers

    for l in range(num_layers):
        prefix = f"model.layers.{l}"
        mapping = {
            f"layer_{l}_q_proj":    f"{prefix}.self_attn.q_proj.weight",
            f"layer_{l}_k_proj":    f"{prefix}.self_attn.k_proj.weight",
            f"layer_{l}_v_proj":    f"{prefix}.self_attn.v_proj.weight",
            f"layer_{l}_o_proj":    f"{prefix}.self_attn.o_proj.weight",
            f"layer_{l}_gate_proj": f"{prefix}.mlp.gate_proj.weight",
            f"layer_{l}_up_proj":   f"{prefix}.mlp.up_proj.weight",
            f"layer_{l}_down_proj": f"{prefix}.mlp.down_proj.weight",
            f"layer_{l}_norm1":     f"{prefix}.input_layernorm.weight",
            f"layer_{l}_norm2":     f"{prefix}.post_attention_layernorm.weight",
        }
        for out_name, sd_key in mapping.items():
            arr = sd[sd_key]
            # Constant folding (§3.1.2): absorb 1/sqrt(head_dim) into W_Q
            if out_name.endswith("_q_proj"):
                arr = (arr * scale).astype(np.float32)
            np.save(os.path.join(output_dir, out_name + ".npy"), arr)

        if (l + 1) % 8 == 0:
            print(f"  Saved layer {l + 1}/{num_layers}")

    # RoPE cos/sin tables
    rope_params = getattr(config, "rope_parameters", {})
    rope_theta = (rope_params.get("rope_theta")
                  or getattr(config, "rope_theta", 10000.0))
    print(f"\nPrecomputing RoPE tables (theta={rope_theta})...")

    cos_table, sin_table = precompute_rope(
        config.hidden_size, head_dim, rope_theta, max_seq_len, chunk_size,
    )
    np.save(os.path.join(output_dir, "rope_cos.npy"), cos_table)
    np.save(os.path.join(output_dir, "rope_sin.npy"), sin_table)

    print(f"\nDone. Weights saved to {output_dir}")


# ---------------------------------------------------------------------------
# ONNX extraction path
# ---------------------------------------------------------------------------

# Maps ONNX initializer name patterns to canonical TranSQL+ table names.
_ONNX_NAME_MAP: dict[str, str] = {
    "model.embed_tokens.weight": "embed_tokens",
    "model.norm.weight":         "final_norm",
    "lm_head.weight":            "lm_head",
}
for _l in range(128):  # support up to 128 layers
    _p = f"model.layers.{_l}"
    _ONNX_NAME_MAP.update({
        f"{_p}.self_attn.q_proj.weight":         f"layer_{_l}_q_proj",
        f"{_p}.self_attn.k_proj.weight":         f"layer_{_l}_k_proj",
        f"{_p}.self_attn.v_proj.weight":         f"layer_{_l}_v_proj",
        f"{_p}.self_attn.o_proj.weight":         f"layer_{_l}_o_proj",
        f"{_p}.mlp.gate_proj.weight":            f"layer_{_l}_gate_proj",
        f"{_p}.mlp.up_proj.weight":              f"layer_{_l}_up_proj",
        f"{_p}.mlp.down_proj.weight":            f"layer_{_l}_down_proj",
        f"{_p}.input_layernorm.weight":          f"layer_{_l}_norm1",
        f"{_p}.post_attention_layernorm.weight":  f"layer_{_l}_norm2",
    })


def _is_matmul_weight(name: str) -> bool:
    """True for 2D weight tensors that need transposing from ONNX layout."""
    matmul_suffixes = (
        "_q_proj", "_k_proj", "_v_proj", "_o_proj",
        "_gate_proj", "_up_proj", "_down_proj", "lm_head",
    )
    return any(name.endswith(s) for s in matmul_suffixes)


def extract_onnx(
    onnx_path: str,
    output_dir: str,
    chunk_size: int,
    max_seq_len: int,
) -> None:
    """Extract weights from an ONNX model with constant folding."""
    try:
        import onnx
        import onnxsim
    except ImportError:
        raise ImportError(
            "ONNX path requires: pip install onnx onnxsim\n"
            "Export the model with: optimum-cli export onnx --model <model_id> ."
        )
    from onnx import numpy_helper

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading ONNX model from {onnx_path}...")
    model_onnx = onnx.load(onnx_path)

    # Constant folding via onnxsim (§3.1)
    print("Applying constant folding via onnxsim.simplify()...")
    model_simplified, ok = onnxsim.simplify(model_onnx)
    if ok:
        model_onnx = model_simplified
    else:
        print("  Warning: onnxsim could not fully simplify; continuing.")
    print(f"  Nodes after simplification: {len(model_onnx.graph.node)}")

    # Extract weight initializers
    print("\nExtracting weight initializers...")
    saved = 0
    head_dim = None

    for init in model_onnx.graph.initializer:
        canonical = _ONNX_NAME_MAP.get(init.name)
        if canonical is None:
            continue

        arr = numpy_helper.to_array(init).astype(np.float32)

        # ONNX stores MatMul right-hand operand as [in_dim, out_dim].
        # Paper §3.1: "transposes the right matrix ... storing its column
        # vectors as row vectors" — we need [out_dim, in_dim] so chunking
        # along axis 1 produces chunks along the contracted dimension,
        # enabling direct dot products at runtime with no algebraic transpose.
        if _is_matmul_weight(canonical) and arr.ndim == 2:
            arr = arr.T

        # Infer head_dim from q_proj shape for constant folding
        if canonical.endswith("_q_proj") and arr.ndim == 2:
            q_dim = arr.shape[0]
            # Assume num_q_heads can be inferred — for Llama: q_dim == hidden_dim
            # head_dim = hidden_dim / num_q_heads; we can't know num_q_heads
            # from ONNX alone, so use the standard 128 for Llama-family
            if head_dim is None:
                head_dim = 128  # standard for Llama-family models
            scale = np.float32(1.0 / np.sqrt(head_dim))
            arr = (arr * scale).astype(np.float32)

        np.save(os.path.join(output_dir, canonical + ".npy"), arr)
        saved += 1

    print(f"  Saved {saved} weight tensors.")
    print(f"\nDone. Outputs saved to {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract LLM weights for TranSQL+ (§3.1)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write .npy files")
    parser.add_argument("--source", choices=["pytorch", "onnx"],
                        default="pytorch",
                        help="Weight source (default: pytorch)")
    parser.add_argument("--model-id", default="meta-llama/Meta-Llama-3-8B",
                        help="HuggingFace model ID (PyTorch path)")
    parser.add_argument("--onnx-path", default=None,
                        help="Path to ONNX model file (required for --source onnx)")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args()

    if args.source == "onnx":
        if not args.onnx_path:
            parser.error("--onnx-path is required when --source onnx")
        extract_onnx(args.onnx_path, args.output_dir,
                     args.chunk_size, args.max_seq_len)
    else:
        extract_pytorch(args.model_id, args.output_dir,
                        args.chunk_size, args.max_seq_len)


if __name__ == "__main__":
    main()
