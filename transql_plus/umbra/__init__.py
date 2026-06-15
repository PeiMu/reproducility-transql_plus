"""
Umbra backend for TranSQL+.

Mirrors the DuckDB-based ``transql_plus`` API (config, DAG, templates,
postopt, runner) but emits PostgreSQL-compatible SQL for the Umbra engine.

Umbra connects via the PostgreSQL wire protocol (port 15432, user postgres).
The Python driver is ``psycopg2``.

Entry point: ``UmbraRunner`` in ``runner_umbra``.
"""
