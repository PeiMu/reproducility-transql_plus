"""
Umbra inference runner for TranSQL+.

Structural mirror of ``transql_plus.runner.TranSQLRunner`` but talks to
Umbra via ``psycopg2`` (PostgreSQL wire protocol, port 15432).

Key dialect differences from DuckDB:

* Every intermediate table is emitted as
  ``CREATE TEMPORARY TABLE t AS (SELECT ...)``. Umbra/PostgreSQL has no
  ``CREATE OR REPLACE TEMP TABLE``; we drop-then-create.
* Temporary tables live for the session (``ON COMMIT PRESERVE ROWS`` is
  the default).
* ``autocommit=True`` avoids wrapping every DDL in a transaction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import psycopg2
import psycopg2.extensions

from ..compute_dag import TensorComputeDAG
from ..config import ModelConfig
from ..postopt import PostOptOptions
from .postopt_umbra import (
    dag_to_sql_umbra,
    iter_pivot_specs,
    pivot_sql,
    postopt_dag_to_sql_umbra,
)
from .sql_templates_umbra import (
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


class UmbraRunner:
    """Inference orchestrator for TranSQL+ on Umbra.

    Public API matches :class:`transql_plus.runner.TranSQLRunner` so the
    benchmark scripts can substitute one for the other.
    """

    def __init__(
        self,
        *,
        config: ModelConfig,
        host: str = "localhost",
        port: int = 15432,
        user: str = "postgres",
        password: str = "umbra",
        database: str = "",
        postopt: PostOptOptions | None = None,
        topology_json: str | None = None,
    ) -> None:
        self.config = config
        self.postopt = postopt
        self._topology_json = topology_json

        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database

        self.con: psycopg2.extensions.connection | None = None
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
        dsn_parts = [
            f"host={self._host}",
            f"port={self._port}",
            f"user={self._user}",
            f"password={self._password}",
        ]
        if self._database:
            dsn_parts.append(f"dbname={self._database}")
        dsn = " ".join(dsn_parts)

        self.con = psycopg2.connect(dsn)
        self.con.set_isolation_level(
            psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT
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
            self._steps = postopt_dag_to_sql_umbra(
                self._dag, self.postopt,
                cached_weight_pivots=(self._pivoted_weights
                                      if self._pivoted_weights else None),
            )
        else:
            self._steps = dag_to_sql_umbra(self._dag)

    def _exec(self, sql: str) -> None:
        with self.con.cursor() as cur:
            cur.execute(sql)

    def _fetchall(self, sql: str) -> list[tuple]:
        with self.con.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()

    def _fetchone(self, sql: str) -> tuple | None:
        with self.con.cursor() as cur:
            cur.execute(sql)
            return cur.fetchone()

    def _create_temp(self, name: str, body_sql: str) -> None:
        """DROP-then-CREATE pattern (no CREATE OR REPLACE in PostgreSQL)."""
        self._exec(f"DROP TABLE IF EXISTS {name}")
        self._exec(
            f"CREATE TEMPORARY TABLE {name} AS ({body_sql})"
        )

    def _pivot_weight_tables(self) -> None:
        """Pre-pivot weight tables once (Decision D9)."""
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

        self._exec("DROP TABLE IF EXISTS input_tokens")
        self._exec(
            "CREATE TEMPORARY TABLE input_tokens "
            "(pos INTEGER, token_id INTEGER)"
        )
        values = ", ".join(f"({i}, {tid})"
                           for i, tid in enumerate(token_ids))
        self._exec(f"INSERT INTO input_tokens VALUES {values}")
        self._temp_tables.append("input_tokens")

        t0 = time.perf_counter()
        for step_idx, (sql, name) in enumerate(self._steps):
            if name in self._pivoted_weights:
                continue
            try:
                self._create_temp(name, sql)
            except Exception as e:
                print(f"\n[UMBRA ERROR] step {step_idx}: {name}")
                print(f"  SQL: {sql[:500]}")
                raise
            self._temp_tables.append(name)
        latency = time.perf_counter() - t0

        return RunResult(latency_s=latency, step_count=len(self._steps))

    def get_output_table(self) -> str:
        if self._dag:
            return self._dag.nodes[self._dag.output_node_id].output_table
        return "logits"

    def get_logits_argmax(self) -> int:
        """Return the argmax token id from the logits table (greedy)."""
        out = self.get_output_table()
        dp_table = out + "_dp"

        if dp_table in set(self._temp_tables):
            row = self._fetchone(
                f"SELECT out_col FROM {dp_table} "
                f"ORDER BY val DESC LIMIT 1"
            )
        else:
            row = self._fetchone(
                f"SELECT chunk_index + _gs.i AS col, v[_gs.i + 1] AS val "
                f"FROM {out}, "
                f"generate_series(0, array_length(v, 1) - 1) AS _gs(i) "
                f"ORDER BY val DESC LIMIT 1"
            )
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # Decode (single-token step with KV cache)
    # ------------------------------------------------------------------

    def run_decode_step(self, token_id: int, pos: int) -> RunResult:
        self._exec("DROP TABLE IF EXISTS dec_input_tokens")
        self._exec(
            "CREATE TEMPORARY TABLE dec_input_tokens "
            "(pos INTEGER, token_id INTEGER)"
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
    # Decode helpers
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
        self._exec_steps(softmax_sql(inp, out, stable=True))

    def _exec_attn_vmul(self, attn: str, v: str, out: str) -> None:
        self._exec_steps(attn_vmul_sql(attn, v, out,
                                       self.config.num_q_heads,
                                       self.config.num_kv_heads,
                                       self.config.head_dim,
                                       self.config.chunk_size))

    def _exec_swiglu(self, gate: str, up: str, out: str) -> None:
        self._exec_steps(swiglu_sql(gate, up, out, self.config.chunk_size))

    def _exec_residual_add(self, a: str, b: str, out: str) -> None:
        self._exec_steps(residual_add_sql(a, b, out, self.config.chunk_size))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _reconnect(self) -> None:
        """Reconnect to Umbra, re-creating pivot tables.

        Umbra has a catalog cache bug where DROP + CREATE of a temp table
        with the same name can lose column metadata.  A fresh connection
        sidesteps this entirely.
        """
        if self.con is not None:
            try:
                self.con.close()
            except Exception:
                pass

        dsn_parts = [
            f"host={self._host}",
            f"port={self._port}",
            f"user={self._user}",
            f"password={self._password}",
        ]
        if self._database:
            dsn_parts.append(f"dbname={self._database}")
        self.con = psycopg2.connect(" ".join(dsn_parts))
        self.con.set_isolation_level(
            psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT
        )

        old_pivots = set(self._pivoted_weights)
        self._pivoted_weights.clear()
        if old_pivots and self.postopt and self.postopt.row2col_pivot:
            self._pivot_weight_tables()

    def _drop_inference_tables(self) -> None:
        """Reset session state for a fresh run.

        Reconnects to Umbra to avoid stale temp-table catalog metadata,
        then re-creates any pre-pivoted weight tables.  Skips reconnect
        on the first call (no tables to clean up).
        """
        if self._temp_tables:
            self._reconnect()
        self._temp_tables.clear()
        self._kv_cache_prepared = False

    def cleanup(self) -> None:
        if self.con is None:
            return
        for t in list(self._temp_tables):
            self._exec(f"DROP TABLE IF EXISTS {t}")
        for t in list(self._pivoted_weights):
            self._exec(f"DROP TABLE IF EXISTS {t}")
        self._temp_tables.clear()
        self._pivoted_weights.clear()

    def close(self) -> None:
        if self.con is not None:
            try:
                self.con.close()
            finally:
                self.con = None
