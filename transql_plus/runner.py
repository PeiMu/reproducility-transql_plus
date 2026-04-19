"""
TranSQL+ inference runner.

Paper reference: Section 5 — evaluation.

Orchestrates the full LLM forward pass inside DuckDB:
  1. Build DAG (from hardcoded builder or topology.json)
  2. Expand to SQL (baseline or post-optimised)
  3. Pre-pivot weight tables (one-time, cached across runs)
  4. Execute prefill or decode steps

Source: AQP_middleware/transql/src/transql_runner.cpp,
        AQP_middleware/measure/run_prefill.py,
        AQP_middleware/measure/run_decode.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from .compute_dag import TensorComputeDAG, TensorOpType
from .config import ModelConfig
from .dag_to_sql import dag_to_sql
from .postopt import (
    PostOptOptions,
    iter_pivot_specs,
    pivot_sql,
    postopt_dag_to_sql,
    _chunk_offsets,
)
from .sql_templates import SqlSteps


@dataclass
class RunResult:
    """Result of an inference run."""
    latency_s: float = 0.0
    step_count: int = 0


class TranSQLRunner:
    """Inference orchestrator for TranSQL+.

    Usage:
        runner = TranSQLRunner(db_path, config)
        runner.init()                     # builds DAG, pivots weights
        result = runner.run_prefill(tokens)  # prefill
        runner.cleanup()                  # drop temp tables
    """

    def __init__(
        self,
        db_path: str,
        config: ModelConfig,
        *,
        postopt: PostOptOptions | None = None,
        read_only: bool = True,
        topology_json: str | None = None,
        memory_limit: str | None = None,
        threads: int | None = None,
        temp_directory: str | None = None,
    ) -> None:
        self.db_path = db_path
        self.config = config
        self.postopt = postopt
        self._read_only = read_only
        self._topology_json = topology_json
        self._memory_limit = memory_limit
        self._threads = threads
        self._temp_directory = temp_directory

        self.con: duckdb.DuckDBPyConnection | None = None
        self._dag: TensorComputeDAG | None = None
        self._steps: SqlSteps = []
        self._pivoted_weights: set[str] = set()  # pivoted TEMP TABLE names
        self._temp_tables: list[str] = []
        self.pivot_setup_time_s: float = 0.0  # one-time D9 pivot cost
        self._kv_cache_prepared: bool = False  # D11: V extracted from fused QKV

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Build DAG, expand to SQL, open connection, pre-pivot weights."""
        import os
        duckdb_config: dict[str, str | int] = {
            "threads": self._threads if self._threads is not None else os.cpu_count(),
        }
        if self._memory_limit is not None:
            duckdb_config["memory_limit"] = self._memory_limit
        if self._temp_directory is not None:
            duckdb_config["temp_directory"] = self._temp_directory
        self.con = duckdb.connect(
            self.db_path,
            read_only=self._read_only,
            config=duckdb_config,
        )

        # Build DAG
        if self._topology_json:
            self._dag = TensorComputeDAG.build_from_json(self._topology_json)
        else:
            self._dag = TensorComputeDAG.build_llama3_8b(self.config)

        # Pre-pivot weight tables (Decision D9) — must happen BEFORE SQL
        # expansion so postopt_dag_to_sql can suppress re-emitting the
        # pivot as a CTE (which would shadow our TEMP TABLE via CTE merge).
        if self.postopt and self.postopt.row2col_pivot:
            pivot_start = time.perf_counter()
            self._pivot_weight_tables()
            self.pivot_setup_time_s = time.perf_counter() - pivot_start

        # Expand to SQL
        if self.postopt:
            self._steps = postopt_dag_to_sql(
                self._dag, self.postopt,
                cached_weight_pivots=(self._pivoted_weights
                                      if self._pivoted_weights else None),
            )
        else:
            self._steps = dag_to_sql(self._dag)

    def _pivot_weight_tables(self) -> None:
        """Pre-pivot weight tables once. Reused across inference runs.

        Decision D9: Weight tables are static — pivoting them once avoids
        repeating this O(n_chunks * n_rows) operation per inference run.
        Uses iter_pivot_specs to enumerate exactly what will be needed
        (respecting table-fusion decisions — Q/K/V and gate/up are fused
        via UNION ALL and never pivoted) and to match the naming that
        pivoted_matmul_sql produces (`_piv` or `_piv_g{N}`).
        """
        for piv_name, weight_table, offsets in iter_pivot_specs(
                self._dag, self.postopt):
            if piv_name in self._pivoted_weights:
                continue
            sql = pivot_sql(weight_table, offsets)
            self.con.execute(f"CREATE TEMP TABLE {piv_name} AS ({sql})")
            self._pivoted_weights.add(piv_name)

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def run_prefill(self, token_ids: list[int]) -> RunResult:
        """Run prefill: process full token sequence through all layers.

        Creates the input_tokens table, executes all SQL steps, returns
        timing result. The output logits are in the last temp table.
        """
        self._drop_inference_tables()

        # Load tokens
        self.con.execute(
            "CREATE TEMP TABLE input_tokens "
            "(pos INTEGER, token_id INTEGER)"
        )
        self.con.executemany(
            "INSERT INTO input_tokens VALUES (?, ?)",
            [(i, tid) for i, tid in enumerate(token_ids)],
        )
        self._temp_tables.append("input_tokens")

        # Execute pipeline. Weight pivots were already filtered out of
        # self._steps by postopt_dag_to_sql(cached_weight_pivots=...);
        # the membership check below is a defensive belt-and-braces.
        t0 = time.perf_counter()
        for sql, name in self._steps:
            if name in self._pivoted_weights:
                continue
            self.con.execute(f"CREATE TEMP TABLE {name} AS ({sql})")
            self._temp_tables.append(name)
        latency = time.perf_counter() - t0

        return RunResult(latency_s=latency, step_count=len(self._steps))

    def get_output_table(self) -> str:
        """Name of the output table after prefill (logits or final node)."""
        if self._dag:
            return self._dag.nodes[self._dag.output_node_id].output_table
        return "logits"

    def get_logits_argmax(self) -> int:
        """Get the argmax token from the logits table (greedy decode)."""
        out = self.get_output_table()
        # Logits are in chunked format — find the max scalar value
        # Use the _dp table if it exists (scalar layout), otherwise unnest
        dp_table = out + "_dp"
        tables = {t for t in self._temp_tables}
        if dp_table in tables:
            result = self.con.execute(
                f"SELECT out_col FROM {dp_table} "
                f"ORDER BY val DESC LIMIT 1"
            ).fetchone()
        else:
            result = self.con.execute(
                f"SELECT row_index * 0 + chunk_index + "
                f"UNNEST(generate_series(0, len(v)-1)) AS global_col, "
                f"UNNEST(v) AS val "
                f"FROM {out} ORDER BY val DESC LIMIT 1"
            ).fetchone()
        return result[0] if result else 0

    # ------------------------------------------------------------------
    # Decode (single-token step with KV cache)
    # ------------------------------------------------------------------

    def run_decode_step(
        self, token_id: int, pos: int,
    ) -> RunResult:
        """Run one autoregressive decode step.

        Processes a single new token through all layers, appends K/V
        to the KV cache tables, and returns timing.

        The KV cache tables (l{l}_k_rope, l{l}_v) must already exist
        from a prior prefill or decode step.

        NOTE: Connection must be opened with read_only=False for decode
        (INSERT INTO requires write access).
        """
        # Create single-token input
        self.con.execute("DROP TABLE IF EXISTS dec_input_tokens")
        self.con.execute(
            "CREATE TEMP TABLE dec_input_tokens "
            "(pos INTEGER, token_id INTEGER)"
        )
        self.con.execute(
            f"INSERT INTO dec_input_tokens VALUES ({pos}, {token_id})"
        )

        # D11: ensure V tables exist for KV-cache INSERT when QKV fusion
        # (§4.2) is active.  Fusion maps l{l}_v to a filter CTE over
        # l{l}_q_qkv, leaving no standalone table for decode to INSERT
        # into.  Runs once after prefill; idempotent on subsequent calls.
        if not self._kv_cache_prepared:
            self._materialize_fused_v_for_decode()
            self._kv_cache_prepared = True

        t0 = time.perf_counter()
        self._run_decode_layers(pos)
        latency = time.perf_counter() - t0

        return RunResult(latency_s=latency)

    def _materialize_fused_v_for_decode(self) -> None:
        """Extract l{l}_v from fused l{l}_q_qkv so decode can INSERT into it.

        See reproduction_note.md D11.  Paper §4.2 (QKV fusion) and §5
        (decode KV cache) do not describe their interaction; fusion
        absorbs the V output table, but decode expects a standalone
        l{l}_v.  This method restores the DAG's shared=True intent for
        V (compute_dag.py marks v with shared=True, noting AttnVMul
        consumption).  Cost: one CTAS per layer of ~seq_len×kv_chunks
        rows (<<1% of prefill time).
        """
        materialised = set(self._temp_tables)
        for l in range(self.config.num_layers):
            v_table = f"l{l}_v"
            qkv_table = f"l{l}_q_qkv"
            if v_table in materialised or qkv_table not in materialised:
                continue
            self.con.execute(
                f"CREATE TEMP TABLE {v_table} AS "
                f"SELECT row_index, chunk_index, v "
                f"FROM {qkv_table} WHERE flag = 'V'"
            )
            self._temp_tables.append(v_table)

    def _run_decode_layers(self, pos: int) -> None:
        """Execute decode-mode SQL for all layers.

        Decode reuses the same SQL templates as prefill — the standard
        templates work correctly for single-row activations. The only
        new operations are INSERT INTO for KV cache extension.
        """
        cs = self.config.chunk_size
        hd = self.config.hidden_dim
        kv = self.config.kv_dim
        ffn = self.config.ffn_dim
        n_hidden = self.config.n_chunks_hidden

        # Embed the new token
        self.con.execute(
            "CREATE OR REPLACE TEMP TABLE dec_x_0 AS ("
            "SELECT t.pos AS row_index, e.chunk_index, e.v "
            "FROM dec_input_tokens t "
            "JOIN embed_tokens e ON t.token_id = e.row_index "
            "ORDER BY t.pos, e.chunk_index)"
        )

        x_in = "dec_x_0"

        for l in range(self.config.num_layers):
            pfx = f"dec_l{l}_"

            def wt(name: str, layer: int = l) -> str:
                return f"layer_{layer}_{name}"

            # RMSNorm1
            norm1 = pfx + "norm1_out"
            self._exec_rmsnorm(x_in, wt("norm1"), norm1)

            # Q, K, V projections
            q = pfx + "q"
            k = pfx + "k"
            v = pfx + "v"
            self._exec_matmul(norm1, wt("q_proj"), q)
            self._exec_matmul(norm1, wt("k_proj"), k)
            self._exec_matmul(norm1, wt("v_proj"), v)

            # RoPE
            q_rope = pfx + "q_rope"
            k_rope = pfx + "k_rope"
            self._exec_rope(q, q_rope)
            self._exec_rope(k, k_rope)

            # Append K/V to cache
            k_cache = f"l{l}_k_rope"
            v_cache = f"l{l}_v"
            self.con.execute(
                f"INSERT INTO {k_cache} "
                f"SELECT row_index, chunk_index, cos, sin "
                f"FROM {k_rope}"
            ) if False else self.con.execute(
                f"INSERT INTO {k_cache} "
                f"SELECT * FROM {k_rope}"
            )
            self.con.execute(
                f"INSERT INTO {v_cache} "
                f"SELECT * FROM {v}"
            )

            # QK attention (single query against full K cache)
            qk = pfx + "qk_scores"
            self._exec_qk_attn(q_rope, k_cache, qk)

            # Softmax
            attn_w = pfx + "attn_weights"
            self._exec_softmax(qk, attn_w)

            # Attention x V (single query against full V cache)
            attn_out = pfx + "attn_out"
            self._exec_attn_vmul(attn_w, v_cache, attn_out)

            # O projection
            o = pfx + "o_proj"
            self._exec_matmul(attn_out, wt("o_proj"), o)

            # Residual add 1
            x_attn = pfx + "x_after_attn"
            self._exec_residual_add(x_in, o, x_attn)

            # RMSNorm2
            norm2 = pfx + "norm2_out"
            self._exec_rmsnorm(x_attn, wt("norm2"), norm2)

            # Gate, Up, SwiGLU, Down
            gate = pfx + "gate"
            up = pfx + "up"
            self._exec_matmul(norm2, wt("gate_proj"), gate)
            self._exec_matmul(norm2, wt("up_proj"), up)

            ffn_act = pfx + "ffn_act"
            self._exec_swiglu(gate, up, ffn_act)

            down = pfx + "down"
            self._exec_matmul(ffn_act, wt("down_proj"), down)

            # Residual add 2
            x_out = pfx + "x_out"
            self._exec_residual_add(x_attn, down, x_out)

            x_in = x_out

        # Final RMSNorm + LM head
        self._exec_rmsnorm(x_in, "final_norm", "dec_final_norm_out")
        self._exec_matmul("dec_final_norm_out", "lm_head", "dec_logits")

    # ------------------------------------------------------------------
    # Decode helpers — execute individual ops via standard SQL templates
    # ------------------------------------------------------------------

    def _exec_step(self, sql: str, name: str) -> None:
        self.con.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS ({sql})")

    def _exec_matmul(self, act: str, weight: str, out: str) -> None:
        from .sql_templates import matmul_sql
        for sql, name in matmul_sql(act, weight, out, self.config.chunk_size):
            self._exec_step(sql, name)

    def _exec_rmsnorm(self, inp: str, gamma: str, out: str) -> None:
        from .sql_templates import rmsnorm_sql
        for sql, name in rmsnorm_sql(inp, gamma, out,
                                      self.config.hidden_dim,
                                      self.config.rms_norm_eps):
            self._exec_step(sql, name)

    def _exec_rope(self, inp: str, out: str) -> None:
        from .sql_templates import rope_sql
        for sql, name in rope_sql(inp, "rope", out, self.config.chunk_size):
            self._exec_step(sql, name)

    def _exec_qk_attn(self, q: str, k: str, out: str) -> None:
        from .sql_templates import qk_attn_sql
        for sql, name in qk_attn_sql(q, k, out,
                                       self.config.num_q_heads,
                                       self.config.num_kv_heads,
                                       self.config.head_dim,
                                       self.config.chunk_size):
            self._exec_step(sql, name)

    def _exec_softmax(self, inp: str, out: str) -> None:
        from .sql_templates import softmax_sql
        for sql, name in softmax_sql(inp, out):
            self._exec_step(sql, name)

    def _exec_attn_vmul(self, attn: str, v: str, out: str) -> None:
        from .sql_templates import attn_vmul_sql
        for sql, name in attn_vmul_sql(attn, v, out,
                                        self.config.num_q_heads,
                                        self.config.num_kv_heads,
                                        self.config.head_dim,
                                        self.config.chunk_size):
            self._exec_step(sql, name)

    def _exec_swiglu(self, gate: str, up: str, out: str) -> None:
        from .sql_templates import swiglu_sql
        for sql, name in swiglu_sql(gate, up, out):
            self._exec_step(sql, name)

    def _exec_residual_add(self, a: str, b: str, out: str) -> None:
        from .sql_templates import residual_add_sql
        for sql, name in residual_add_sql(a, b, out):
            self._exec_step(sql, name)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _drop_inference_tables(self) -> None:
        """Drop temp tables from a previous inference run.

        Preserves cached weight pivot tables.
        """
        tables = self.con.execute(
            "SELECT table_name FROM duckdb_tables() WHERE temporary = true"
        ).fetchall()
        for (t,) in tables:
            if t in self._pivoted_weights:
                continue
            # Keep KV cache tables for decode
            self.con.execute(f'DROP TABLE IF EXISTS "{t}"')
        self._temp_tables.clear()
        self._kv_cache_prepared = False

    def cleanup(self) -> None:
        """Drop all temp tables including weight pivots."""
        if self.con:
            tables = self.con.execute(
                "SELECT table_name FROM duckdb_tables() "
                "WHERE temporary = true"
            ).fetchall()
            for (t,) in tables:
                self.con.execute(f'DROP TABLE IF EXISTS "{t}"')
            self._temp_tables.clear()
            self._pivoted_weights.clear()

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self.con:
            self.con.close()
            self.con = None
