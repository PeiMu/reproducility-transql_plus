"""
ClickHouse inference runner for TranSQL+.

Structural mirror of ``transql_plus.runner.TranSQLRunner`` but talks to
ClickHouse via ``clickhouse_connect`` (HTTP, port 8123).

Key dialect differences (see reproduction_note.md D12):

* Every intermediate table is emitted as
  ``CREATE TEMPORARY TABLE t ENGINE=Memory AS SELECT ...`` (D12: no
  ``CREATE OR REPLACE TEMP TABLE``; we drop-then-create).
* Session scoping keeps temporary tables alive across per-step SQL
  requests — we set ``session_id`` once on the client instance.
* Catalogue lookups go through ``system.tables WHERE is_temporary = 1``
  instead of DuckDB's ``duckdb_tables()``.
* Resource settings match the paper's c7.2xlarge profile:
  ``max_memory_usage`` and ``max_threads`` apply per query.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import clickhouse_connect

from ..compute_dag import TensorComputeDAG
from ..config import ModelConfig
from ..postopt import PostOptOptions
from .postopt_ch import (
    dag_to_sql_ch,
    iter_pivot_specs,
    pivot_sql,
    postopt_dag_to_sql_ch,
)
from .sql_templates_ch import (
    SqlSteps,
    attn_vmul_sql,
    matmul_sql,
    qk_attn_sql,
    residual_add_sql,
    rmsnorm_sql,
    rope_sql,
    softmax_sql,
    swiglu_sql,
)


@dataclass
class RunResult:
    latency_s: float = 0.0
    step_count: int = 0


class ClickHouseRunner:
    """Inference orchestrator for TranSQL+ on ClickHouse.

    Public API matches :class:`transql_plus.runner.TranSQLRunner` so the
    benchmark scripts can substitute one for the other.
    """

    def __init__(
        self,
        *,
        config: ModelConfig,
        host: str = "localhost",
        port: int = 8123,
        user: str = "default",
        password: str = "",
        database: str = "default",
        postopt: PostOptOptions | None = None,
        topology_json: str | None = None,
        max_memory_usage: int | None = None,   # bytes
        max_threads: int | None = None,
        session_id: str | None = None,
    ) -> None:
        self.config = config
        self.postopt = postopt
        self._topology_json = topology_json

        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database
        self._max_memory_usage = max_memory_usage
        self._max_threads = max_threads
        self._session_id = session_id or f"transql_ch_{uuid.uuid4().hex[:12]}"

        self.client: clickhouse_connect.driver.client.Client | None = None
        self._dag: TensorComputeDAG | None = None
        self._steps: SqlSteps = []
        self._pivoted_weights: set[str] = set()
        self._temp_tables: list[str] = []
        self.pivot_setup_time_s: float = 0.0
        self._kv_cache_prepared: bool = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init(self) -> None:
        settings: dict[str, int] = {}
        if self._max_memory_usage is not None:
            settings["max_memory_usage"] = self._max_memory_usage
        if self._max_threads is not None:
            settings["max_threads"] = self._max_threads

        self.client = clickhouse_connect.get_client(
            host=self._host, port=self._port,
            username=self._user, password=self._password,
            database=self._database,
            session_id=self._session_id,
            settings=settings,
        )

        if self._topology_json:
            self._dag = TensorComputeDAG.build_from_json(self._topology_json)
        else:
            self._dag = TensorComputeDAG.build_llama3_8b(self.config)

        if self.postopt and self.postopt.row2col_pivot:
            pivot_start = time.perf_counter()
            self._pivot_weight_tables()
            self.pivot_setup_time_s = time.perf_counter() - pivot_start

        if self.postopt:
            self._steps = postopt_dag_to_sql_ch(
                self._dag, self.postopt,
                cached_weight_pivots=(self._pivoted_weights
                                      if self._pivoted_weights else None),
            )
        else:
            self._steps = dag_to_sql_ch(self._dag)

    def _exec(self, sql: str) -> None:
        """Execute a statement with no result payload."""
        assert self.client is not None
        self.client.command(sql)

    def _create_temp(self, name: str, body_sql: str) -> None:
        """``CREATE OR REPLACE TEMP TABLE`` equivalent for ClickHouse.

        ClickHouse has no ``OR REPLACE``; drop then create.
        """
        assert self.client is not None
        self._exec(f"DROP TEMPORARY TABLE IF EXISTS {name}")
        self._exec(
            f"CREATE TEMPORARY TABLE {name} ENGINE=Memory AS ({body_sql})"
        )

    def _pivot_weight_tables(self) -> None:
        """Pre-pivot weight tables once (Decision D9).

        Uses the ClickHouse variant of ``iter_pivot_specs`` so fusion is
        respected (fused Q/K/V and gate/up weights are not pivoted).
        """
        for piv_name, weight_table, offsets in iter_pivot_specs(
                self._dag, self.postopt):
            if piv_name in self._pivoted_weights:
                continue
            sql = pivot_sql(weight_table, offsets)
            self._create_temp(piv_name, sql)
            self._pivoted_weights.add(piv_name)

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def run_prefill(self, token_ids: list[int]) -> RunResult:
        self._drop_inference_tables()

        # Load tokens. ClickHouse has no parameterised executemany(?,?);
        # build a VALUES literal. Prompts are short (<= 200 tokens).
        self._exec("DROP TEMPORARY TABLE IF EXISTS input_tokens")
        self._exec(
            "CREATE TEMPORARY TABLE input_tokens "
            "(pos Int32, token_id Int32) ENGINE=Memory"
        )
        values = ", ".join(f"({i}, {tid})"
                           for i, tid in enumerate(token_ids))
        self._exec(f"INSERT INTO input_tokens VALUES {values}")
        self._temp_tables.append("input_tokens")

        t0 = time.perf_counter()
        for sql, name in self._steps:
            if name in self._pivoted_weights:
                continue
            self._create_temp(name, sql)
            self._temp_tables.append(name)
        latency = time.perf_counter() - t0

        return RunResult(latency_s=latency, step_count=len(self._steps))

    def get_output_table(self) -> str:
        if self._dag:
            return self._dag.nodes[self._dag.output_node_id].output_table
        return "logits"

    def get_logits_argmax(self) -> int:
        """Return the argmax token id from the logits table (greedy)."""
        assert self.client is not None
        out = self.get_output_table()
        dp_table = out + "_dp"

        if dp_table in set(self._temp_tables):
            rows = self.client.query(
                f"SELECT out_col FROM {dp_table} "
                f"ORDER BY val DESC LIMIT 1"
            ).result_rows
        else:
            # Scalarise the chunked output and take the global argmax.
            # arrayJoin over v paired with arrayEnumerate gives per-element
            # (local_pos, value) tuples; add chunk_index for the global col.
            rows = self.client.query(
                f"SELECT row_index * 0 + chunk_index + _ep.1 - 1 AS col, "
                f"_ep.2 AS val FROM ("
                f"SELECT row_index, chunk_index, "
                f"arrayJoin(arrayZip(arrayEnumerate(v), v)) AS _ep "
                f"FROM {out}"
                f") ORDER BY val DESC LIMIT 1"
            ).result_rows
        return int(rows[0][0]) if rows else 0

    # ------------------------------------------------------------------
    # Decode (single-token step with KV cache)
    # ------------------------------------------------------------------

    def run_decode_step(self, token_id: int, pos: int) -> RunResult:
        self._exec("DROP TEMPORARY TABLE IF EXISTS dec_input_tokens")
        self._exec(
            "CREATE TEMPORARY TABLE dec_input_tokens "
            "(pos Int32, token_id Int32) ENGINE=Memory"
        )
        self._exec(
            f"INSERT INTO dec_input_tokens VALUES ({pos}, {token_id})"
        )

        if not self._kv_cache_prepared:
            self._materialize_fused_v_for_decode()
            self._kv_cache_prepared = True

        t0 = time.perf_counter()
        self._run_decode_layers(pos)
        latency = time.perf_counter() - t0
        return RunResult(latency_s=latency)

    def _materialize_fused_v_for_decode(self) -> None:
        """D11: extract l{l}_v from fused l{l}_q_qkv for KV-cache INSERT."""
        materialised = set(self._temp_tables)
        for l in range(self.config.num_layers):
            v_table = f"l{l}_v"
            qkv_table = f"l{l}_q_qkv"
            if v_table in materialised or qkv_table not in materialised:
                continue
            self._create_temp(
                v_table,
                f"SELECT row_index, chunk_index, v "
                f"FROM {qkv_table} WHERE flag = 'V'",
            )
            self._temp_tables.append(v_table)

    def _run_decode_layers(self, pos: int) -> None:
        for l in range(self.config.num_layers):
            pfx = f"dec_l{l}_"

            def wt(name: str, layer: int = l) -> str:
                return f"layer_{layer}_{name}"

            # Embed the new token on first layer
            if l == 0:
                self._create_temp(
                    "dec_x_0",
                    "SELECT t.pos AS row_index, e.chunk_index, e.v "
                    "FROM dec_input_tokens t "
                    "JOIN embed_tokens e ON t.token_id = e.row_index "
                    "ORDER BY t.pos, e.chunk_index",
                )
            x_in = f"dec_l{l-1}_x_out" if l > 0 else "dec_x_0"

            norm1 = pfx + "norm1_out"
            self._exec_rmsnorm(x_in, wt("norm1"), norm1)

            q = pfx + "q"
            k = pfx + "k"
            v = pfx + "v"
            self._exec_matmul(norm1, wt("q_proj"), q)
            self._exec_matmul(norm1, wt("k_proj"), k)
            self._exec_matmul(norm1, wt("v_proj"), v)

            q_rope = pfx + "q_rope"
            k_rope = pfx + "k_rope"
            self._exec_rope(q, q_rope)
            self._exec_rope(k, k_rope)

            k_cache = f"l{l}_k_rope"
            v_cache = f"l{l}_v"
            self._exec(f"INSERT INTO {k_cache} SELECT * FROM {k_rope}")
            self._exec(f"INSERT INTO {v_cache} SELECT * FROM {v}")

            qk = pfx + "qk_scores"
            self._exec_qk_attn(q_rope, k_cache, qk)

            attn_w = pfx + "attn_weights"
            self._exec_softmax(qk, attn_w)

            attn_out = pfx + "attn_out"
            self._exec_attn_vmul(attn_w, v_cache, attn_out)

            o = pfx + "o_proj"
            self._exec_matmul(attn_out, wt("o_proj"), o)

            x_attn = pfx + "x_after_attn"
            self._exec_residual_add(x_in, o, x_attn)

            norm2 = pfx + "norm2_out"
            self._exec_rmsnorm(x_attn, wt("norm2"), norm2)

            gate = pfx + "gate"
            up = pfx + "up"
            self._exec_matmul(norm2, wt("gate_proj"), gate)
            self._exec_matmul(norm2, wt("up_proj"), up)

            ffn_act = pfx + "ffn_act"
            self._exec_swiglu(gate, up, ffn_act)

            down = pfx + "down"
            self._exec_matmul(ffn_act, wt("down_proj"), down)

            x_out = pfx + "x_out"
            self._exec_residual_add(x_attn, down, x_out)

        self._exec_rmsnorm(f"dec_l{self.config.num_layers-1}_x_out",
                           "final_norm", "dec_final_norm_out")
        self._exec_matmul("dec_final_norm_out", "lm_head", "dec_logits")

    # ------------------------------------------------------------------
    # Decode helpers — re-use the template library
    # ------------------------------------------------------------------

    def _exec_steps(self, steps: SqlSteps) -> None:
        for sql, name in steps:
            self._create_temp(name, sql)

    def _exec_matmul(self, act: str, weight: str, out: str) -> None:
        self._exec_steps(matmul_sql(act, weight, out, self.config.chunk_size))

    def _exec_rmsnorm(self, inp: str, gamma: str, out: str) -> None:
        self._exec_steps(rmsnorm_sql(inp, gamma, out,
                                     self.config.hidden_dim,
                                     self.config.rms_norm_eps))

    def _exec_rope(self, inp: str, out: str) -> None:
        self._exec_steps(rope_sql(inp, "rope", out, self.config.chunk_size))

    def _exec_qk_attn(self, q: str, k: str, out: str) -> None:
        self._exec_steps(qk_attn_sql(q, k, out,
                                     self.config.num_q_heads,
                                     self.config.num_kv_heads,
                                     self.config.head_dim,
                                     self.config.chunk_size))

    def _exec_softmax(self, inp: str, out: str) -> None:
        self._exec_steps(softmax_sql(inp, out))

    def _exec_attn_vmul(self, attn: str, v: str, out: str) -> None:
        self._exec_steps(attn_vmul_sql(attn, v, out,
                                       self.config.num_q_heads,
                                       self.config.num_kv_heads,
                                       self.config.head_dim,
                                       self.config.chunk_size))

    def _exec_swiglu(self, gate: str, up: str, out: str) -> None:
        self._exec_steps(swiglu_sql(gate, up, out))

    def _exec_residual_add(self, a: str, b: str, out: str) -> None:
        self._exec_steps(residual_add_sql(a, b, out))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _drop_inference_tables(self) -> None:
        """Drop temp tables from a previous run, preserving weight pivots."""
        assert self.client is not None
        rows = self.client.query(
            "SELECT name FROM system.tables WHERE is_temporary = 1"
        ).result_rows
        for (name,) in rows:
            if name in self._pivoted_weights:
                continue
            self._exec(f"DROP TEMPORARY TABLE IF EXISTS {name}")
        self._temp_tables.clear()
        self._kv_cache_prepared = False

    def cleanup(self) -> None:
        if self.client is None:
            return
        rows = self.client.query(
            "SELECT name FROM system.tables WHERE is_temporary = 1"
        ).result_rows
        for (name,) in rows:
            self._exec(f"DROP TEMPORARY TABLE IF EXISTS {name}")
        self._temp_tables.clear()
        self._pivoted_weights.clear()

    def close(self) -> None:
        if self.client is not None:
            try:
                self.client.close()
            finally:
                self.client = None
