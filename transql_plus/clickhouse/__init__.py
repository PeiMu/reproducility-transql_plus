"""
ClickHouse backend for TranSQL+.

Mirrors the DuckDB-based ``transql_plus`` API (config, DAG, templates,
postopt, runner) but emits ClickHouse SQL. See ``reproduction_note.md``
section D12 for the dialect gap table and ``results/clickhouse_sql_probe.json``
for the raw probe output.

Entry point: ``ClickHouseRunner`` in ``runner_ch``.
"""
