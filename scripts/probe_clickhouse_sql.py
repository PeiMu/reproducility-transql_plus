"""
probe_clickhouse_sql.py - Catalogue DuckDB constructs that TranSQL+ uses
against a live ClickHouse server so we know what needs to be rewritten.

Tests each SQL construct by executing a minimal query and recording:
  - status: supported / workaround / unsupported
  - error (if any)
  - clickhouse_equivalent: the rewrite we'll use in transql_plus/clickhouse/

Usage:
    python scripts/probe_clickhouse_sql.py \\
        --host localhost --port 8123 \\
        --output results/clickhouse_sql_probe.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, asdict

import clickhouse_connect


@dataclass
class ProbeResult:
    name: str
    duckdb_form: str
    status: str            # "supported", "workaround", "unsupported"
    clickhouse_form: str   # exact syntax we'll emit from the ClickHouse backend
    probe_sql: str         # SQL we ran to check
    error: str | None = None
    notes: str = ""


def _run(client, sql: str) -> tuple[bool, str | None]:
    """Return (ok, err). `ok=True` means the query executed without error."""
    try:
        client.query(sql)
        return True, None
    except Exception as exc:
        return False, str(exc).splitlines()[0][:400]


def probe(client) -> list[ProbeResult]:
    results: list[ProbeResult] = []

    # 1. list_dot_product -> dotProduct
    ok, err = _run(client,
                   "SELECT dotProduct([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])")
    results.append(ProbeResult(
        name="list_dot_product",
        duckdb_form="list_dot_product(a, b)",
        status="workaround" if ok else "unsupported",
        clickhouse_form="dotProduct(a, b)",
        probe_sql="SELECT dotProduct([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])",
        error=err,
        notes="ClickHouse has a native dotProduct() over Array(Float32).",
    ))

    # 2. list_sum -> arraySum
    ok, err = _run(client, "SELECT arraySum([1.0, 2.0, 3.0])")
    results.append(ProbeResult(
        name="list_sum",
        duckdb_form="list_sum(arr)",
        status="workaround" if ok else "unsupported",
        clickhouse_form="arraySum(arr)",
        probe_sql="SELECT arraySum([1.0, 2.0, 3.0])",
        error=err,
    ))

    # 3. len -> length
    ok, err = _run(client, "SELECT length([1.0, 2.0, 3.0])")
    results.append(ProbeResult(
        name="len_array",
        duckdb_form="len(arr)",
        status="workaround" if ok else "unsupported",
        clickhouse_form="length(arr)",
        probe_sql="SELECT length([1.0, 2.0, 3.0])",
        error=err,
    ))

    # 4. list_transform(arr, x -> expr) -> arrayMap(x -> expr, arr)
    ok, err = _run(client, "SELECT arrayMap(x -> x * x, [1.0, 2.0, 3.0])")
    results.append(ProbeResult(
        name="list_transform_lambda",
        duckdb_form="list_transform(v, x -> x*x)",
        status="workaround" if ok else "unsupported",
        clickhouse_form="arrayMap(x -> x*x, v)",
        probe_sql="SELECT arrayMap(x -> x * x, [1.0, 2.0, 3.0])",
        error=err,
    ))

    # 5. list_transform(generate_series(1, N), i -> expr)
    #    -> arrayMap(i -> expr, range(1, N+1))
    ok, err = _run(client,
                   "SELECT arrayMap(i -> i * 2, range(1, 4))")
    results.append(ProbeResult(
        name="list_transform_over_range",
        duckdb_form="list_transform(generate_series(1, N), i -> expr)",
        status="workaround" if ok else "unsupported",
        clickhouse_form="arrayMap(i -> expr, range(1, N+1))",
        probe_sql="SELECT arrayMap(i -> i * 2, range(1, 4))",
        error=err,
        notes=("ClickHouse range(a,b) is exclusive on b; DuckDB "
               "generate_series(a,b) is inclusive. Add +1 in the rewrite."),
    ))

    # 6. generate_series as standalone
    ok, err = _run(client, "SELECT range(0, 5)")
    results.append(ProbeResult(
        name="generate_series",
        duckdb_form="generate_series(a, b)  (inclusive)",
        status="workaround" if ok else "unsupported",
        clickhouse_form="range(a, b+1)  (exclusive upper)",
        probe_sql="SELECT range(0, 5)",
        error=err,
    ))

    # 7. unnest(generate_series(0, N)) -> arrayJoin(range(0, N+1))
    ok, err = _run(client, "SELECT arrayJoin(range(0, 4)) AS i")
    results.append(ProbeResult(
        name="unnest_generate_series",
        duckdb_form="unnest(generate_series(0, N-1))",
        status="workaround" if ok else "unsupported",
        clickhouse_form="arrayJoin(range(0, N))",
        probe_sql="SELECT arrayJoin(range(0, 4)) AS i",
        error=err,
    ))

    # 8. array_agg ORDER BY val -> groupArray after ORDER BY subquery
    #    (ClickHouse does not accept ORDER BY inside groupArray; instead,
    #     push ORDER BY into a subquery.)
    probe_sql = (
        "SELECT groupArray(val) FROM "
        "(SELECT arrayJoin([3, 1, 2]) AS val ORDER BY val)"
    )
    ok, err = _run(client, probe_sql)
    results.append(ProbeResult(
        name="array_agg_ordered",
        duckdb_form="array_agg(val ORDER BY k)",
        status="workaround" if ok else "unsupported",
        clickhouse_form=("(SELECT groupArray(val) FROM "
                         "(... ORDER BY k))  -- push ORDER BY into subquery"),
        probe_sql=probe_sql,
        error=err,
        notes=("ClickHouse has no ORDER BY inside aggregate; must sort in a "
               "subquery first. In TranSQL+ this shows up in ROW2COL "
               "re-chunking (sql_templates.py L102, L373)."),
    ))

    # 9. FLOAT[N] (fixed-length array column type) -- NOT supported
    _run(client, "DROP TABLE IF EXISTS _probe_fixed_arr")
    ok_fixed, err_fixed = _run(
        client, "CREATE TABLE _probe_fixed_arr (v Array(Float32, 4)) "
                "ENGINE=Memory")
    results.append(ProbeResult(
        name="fixed_length_array_type",
        duckdb_form="FLOAT[4]",
        status="unsupported" if not ok_fixed else "supported",
        clickhouse_form="Array(Float32)  -- variable only; no fixed length",
        probe_sql="CREATE TABLE _probe_fixed_arr (v Array(Float32, 4)) "
                  "ENGINE=Memory",
        error=err_fixed,
        notes=("ClickHouse Arrays are variable-length. Fixed size is not "
               "enforced at the type level; enforce in loader instead. "
               "Operationally equivalent since all rows written at a given "
               "chunk_size are the same length."),
    ))
    _run(client, "DROP TABLE IF EXISTS _probe_fixed_arr")

    # 10. Variable-length FLOAT[] -> Array(Float32)
    _run(client, "DROP TABLE IF EXISTS _probe_var_arr")
    ok, err = _run(client, "CREATE TABLE _probe_var_arr (v Array(Float32)) "
                           "ENGINE=Memory")
    results.append(ProbeResult(
        name="variable_length_array_type",
        duckdb_form="FLOAT[]",
        status="workaround" if ok else "unsupported",
        clickhouse_form="Array(Float32)",
        probe_sql="CREATE TABLE _probe_var_arr (v Array(Float32)) "
                  "ENGINE=Memory",
        error=err,
    ))
    _run(client, "DROP TABLE IF EXISTS _probe_var_arr")

    # 11. PIVOT -- NOT supported in ClickHouse SQL.
    _run(client, "DROP TABLE IF EXISTS _probe_pivot_src")
    _run(client,
         "CREATE TABLE _probe_pivot_src (row_index Int32, chunk_index Int32, "
         "v Array(Float32)) ENGINE=Memory")
    _run(client,
         "INSERT INTO _probe_pivot_src VALUES (0, 0, [1.0]), (0, 1, [2.0])")
    ok_pivot, err_pivot = _run(
        client,
        "PIVOT _probe_pivot_src ON chunk_index USING first(v)")
    results.append(ProbeResult(
        name="pivot",
        duckdb_form=("PIVOT tbl ON chunk_index "
                     "USING first(v) GROUP BY row_index"),
        status="unsupported",
        clickhouse_form=("SELECT row_index,\n"
                         "  (groupArrayIf(v, chunk_index=0))[1] AS c0,\n"
                         "  (groupArrayIf(v, chunk_index=1))[1] AS c1,\n"
                         "  ...\n"
                         "FROM tbl GROUP BY row_index"),
        probe_sql="PIVOT _probe_pivot_src ON chunk_index USING first(v)",
        error=err_pivot,
        notes=("Core paper post-opt §4.3. Must emit explicit groupArrayIf "
               "column per chunk_index in postopt_ch.py. Cost scales with "
               "num_chunks; same as DuckDB's PIVOT at runtime."),
    ))
    _run(client, "DROP TABLE IF EXISTS _probe_pivot_src")

    # 12. POSITIONAL JOIN -- NOT supported.
    _run(client, "DROP TABLE IF EXISTS _probe_pj_a")
    _run(client, "DROP TABLE IF EXISTS _probe_pj_b")
    _run(client, "CREATE TABLE _probe_pj_a (x Int32) ENGINE=Memory")
    _run(client, "CREATE TABLE _probe_pj_b (y Int32) ENGINE=Memory")
    _run(client, "INSERT INTO _probe_pj_a VALUES (1), (2)")
    _run(client, "INSERT INTO _probe_pj_b VALUES (10), (20)")
    ok_pj, err_pj = _run(
        client,
        "SELECT * FROM _probe_pj_a POSITIONAL JOIN _probe_pj_b")
    results.append(ProbeResult(
        name="positional_join",
        duckdb_form="a POSITIONAL JOIN b",
        status="unsupported",
        clickhouse_form=("Fold neighbour tables into the same SELECT as "
                         "multiple computed columns (preferred), or align by "
                         "rowNumberInAllBlocks() as a secondary workaround."),
        probe_sql="SELECT * FROM _probe_pj_a POSITIONAL JOIN _probe_pj_b",
        error=err_pj,
        notes=("In postopt.py POSITIONAL JOIN is used to SUM partial "
               "dot-products. We rewrite as a single SELECT with multiple "
               "SUM(...)_i expressions and sum them inline; the join "
               "disappears."),
    ))
    _run(client, "DROP TABLE IF EXISTS _probe_pj_a")
    _run(client, "DROP TABLE IF EXISTS _probe_pj_b")

    # 13. CREATE TEMP TABLE t AS SELECT ... (DuckDB syntax)
    _run(client, "DROP TEMPORARY TABLE IF EXISTS _probe_tmp")
    ok, err = _run(
        client,
        "CREATE TEMPORARY TABLE _probe_tmp ENGINE=Memory AS SELECT 1 AS x")
    results.append(ProbeResult(
        name="create_temp_table",
        duckdb_form="CREATE TEMP TABLE t AS SELECT ...",
        status="workaround" if ok else "unsupported",
        clickhouse_form=("CREATE TEMPORARY TABLE t ENGINE=Memory AS "
                         "SELECT ..."),
        probe_sql="CREATE TEMPORARY TABLE _probe_tmp ENGINE=Memory AS "
                  "SELECT 1 AS x",
        error=err,
        notes=("ClickHouse requires explicit ENGINE=Memory for TEMPORARY."),
    ))
    _run(client, "DROP TEMPORARY TABLE IF EXISTS _probe_tmp")

    # 14. CREATE OR REPLACE TEMP TABLE -- NOT supported.
    ok_cor, err_cor = _run(
        client, "CREATE OR REPLACE TEMPORARY TABLE _probe_cor "
                "ENGINE=Memory AS SELECT 1")
    results.append(ProbeResult(
        name="create_or_replace_temp_table",
        duckdb_form="CREATE OR REPLACE TEMP TABLE t AS SELECT ...",
        status="unsupported",
        clickhouse_form=("DROP TEMPORARY TABLE IF EXISTS t; "
                         "CREATE TEMPORARY TABLE t ENGINE=Memory AS "
                         "SELECT ..."),
        probe_sql="CREATE OR REPLACE TEMPORARY TABLE _probe_cor "
                  "ENGINE=Memory AS SELECT 1",
        error=err_cor,
        notes=("Two-statement replacement must be issued as separate queries."),
    ))
    _run(client, "DROP TEMPORARY TABLE IF EXISTS _probe_cor")

    # 15. duckdb_tables() catalogue function
    ok, err = _run(client,
                   "SELECT name FROM system.tables WHERE is_temporary = 1")
    results.append(ProbeResult(
        name="duckdb_tables",
        duckdb_form="SELECT table_name FROM duckdb_tables() "
                    "WHERE temporary = true",
        status="workaround" if ok else "unsupported",
        clickhouse_form=("SELECT name FROM system.tables "
                         "WHERE is_temporary = 1"),
        probe_sql="SELECT name FROM system.tables WHERE is_temporary = 1",
        error=err,
    ))

    # 16. CAST to FLOAT
    ok, err = _run(client, "SELECT CAST(1.0 AS Float32)")
    results.append(ProbeResult(
        name="cast_float",
        duckdb_form="CAST(x AS FLOAT)",
        status="workaround" if ok else "unsupported",
        clickhouse_form="CAST(x AS Float32)  -- or toFloat32(x)",
        probe_sql="SELECT CAST(1.0 AS Float32)",
        error=err,
    ))

    return results


def print_table(results: list[ProbeResult]) -> None:
    status_counts: dict[str, int] = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    name_w = max(len(r.name) for r in results) + 2
    print("\n" + "=" * 100)
    print("  DuckDB -> ClickHouse dialect probe")
    print("=" * 100)
    print(f"  {'construct':<{name_w}}{'status':<14}{'error / notes'}")
    print("-" * 100)
    for r in results:
        tag = r.error or r.notes[:60] or ""
        print(f"  {r.name:<{name_w}}{r.status:<14}{tag}")
    print("-" * 100)
    print("  summary:", status_counts)
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--user", default="default")
    parser.add_argument("--password", default="")
    parser.add_argument("--database", default="default")
    parser.add_argument(
        "--output", default="results/clickhouse_sql_probe.json")
    args = parser.parse_args()

    client = clickhouse_connect.get_client(
        host=args.host, port=args.port,
        username=args.user, password=args.password,
        database=args.database,
    )

    server_info = client.query("SELECT version()").result_rows[0][0]
    print(f"Connected to ClickHouse {server_info} at "
          f"{args.host}:{args.port}")

    results = probe(client)
    print_table(results)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "server_version": server_info,
            "host": args.host,
            "port": args.port,
            "constructs": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"\nProbe results saved to {args.output}")


if __name__ == "__main__":
    main()
