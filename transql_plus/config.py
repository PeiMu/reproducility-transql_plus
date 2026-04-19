"""
Model configuration for TranSQL+.

Paper reference: Section 3.1 (model preprocessing), Section 5 (Llama3-8B parameters).

All tensor dimensions flow through ModelConfig — no hardcoded magic numbers
in SQL templates or post-optimisation code (Decision D6).
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    hidden_dim: int       # 4096 for Llama3-8B
    num_q_heads: int      # 32
    num_kv_heads: int     # 8
    head_dim: int         # 128  (hidden_dim / num_q_heads)
    ffn_dim: int          # 14336 (intermediate_size)
    num_layers: int       # 32
    vocab_size: int       # 128256
    rms_norm_eps: float   # 1e-5
    rope_theta: float     # 500000.0 for Llama3; 10000.0 for Llama2
    max_seq_len: int      # 2048
    chunk_size: int       # 32

    # -- derived properties ------------------------------------------------

    @property
    def kv_dim(self) -> int:
        """Total key/value projection dimension = num_kv_heads * head_dim."""
        return self.num_kv_heads * self.head_dim

    @property
    def n_chunks_hidden(self) -> int:
        """Number of chunks along hidden_dim."""
        return self.hidden_dim // self.chunk_size

    @property
    def n_chunks_ffn(self) -> int:
        """Number of chunks along ffn_dim."""
        return self.ffn_dim // self.chunk_size

    @property
    def chunks_per_head(self) -> int:
        """Chunks per attention head = head_dim / chunk_size."""
        return self.head_dim // self.chunk_size

    @property
    def group_size(self) -> int:
        """GQA group size = num_q_heads / num_kv_heads."""
        return self.num_q_heads // self.num_kv_heads

    @property
    def half_chunk(self) -> int:
        """Half chunk size, used for RoPE cos/sin pairs."""
        return self.chunk_size // 2

    # -- factory methods ---------------------------------------------------

    @classmethod
    def llama3_8b(cls, chunk_size: int = 32) -> "ModelConfig":
        return cls(
            hidden_dim=4096,
            num_q_heads=32,
            num_kv_heads=8,
            head_dim=128,
            ffn_dim=14336,
            num_layers=32,
            vocab_size=128256,
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
            max_seq_len=2048,
            chunk_size=chunk_size,
        )
