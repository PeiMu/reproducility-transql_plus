"""
Measure TranSQL+ perplexity on WikiText-2 using ClickHouse.

Companion to ``scripts/run_perplexity.py`` (DuckDB). Cross-entropy is
computed in SQL to avoid fetching ``seq_len * vocab_size`` floats back
to Python.

Dialect shift vs. the DuckDB script (see reproduction_note.md D12):
DuckDB's paired ``UNNEST(generate_series(0, len(v)-1)) ... UNNEST(v)``
becomes ``arrayJoin(arrayZip(arrayEnumerate(v), v))`` in ClickHouse
(positions and values share a tuple so they stay aligned).

Usage:
    python scripts/run_clickhouse_perplexity.py \\
        --ch-host localhost --ch-port 8123 --ch-database default \\
        --output results/clickhouse_ppl.json \\
        [--num-layers 32] [--max-chunks 64] [--num-workers 1]

Requires: ``pip install datasets transformers clickhouse-connect``.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import time

from transql_plus.clickhouse.runner_ch import ClickHouseRunner
from transql_plus.config import ModelConfig
from transql_plus.postopt import PostOptOptions


MODEL_ID = "meta-llama/Meta-Llama-3-8B"
# Same default as the DuckDB script: 512-token evaluation chunks.
DEFAULT_CONTEXT_LEN = 512


def parse_bytes(s: str) -> int:
    """``"16GB"`` → bytes. 1024-based (GB == GiB, matches DuckDB)."""
    s = s.strip().upper()
    mults = [
        ("TIB", 1024 ** 4), ("GIB", 1024 ** 3),
        ("MIB", 1024 ** 2), ("KIB", 1024),
        ("TB",  1024 ** 4), ("GB",  1024 ** 3),
        ("MB",  1024 ** 2), ("KB",  1024),
        ("T",   1024 ** 4), ("G",   1024 ** 3),
        ("M",   1024 ** 2), ("K",   1024),
        ("B",   1),
    ]
    for suf, m in mults:
        if s.endswith(suf):
            return int(float(s[:-len(suf)]) * m)
    return int(s)


def _run_chunk(args: tuple) -> tuple[int, float, int, float, float]:
    """Worker: run prefill on one chunk and compute cross-entropy in SQL.

    Returns ``(chunk_idx, chunk_loss, chunk_tokens, dt, pivot_setup_s)``.
    """
    (host, port, user, password, database,
     token_ids, num_layers, chunk_size, chunk_idx,
     max_memory_usage, max_threads,
     use_pivot, pivot_width, subquery_width) = args

    config = ModelConfig.llama3_8b(chunk_size=chunk_size)
    config = ModelConfig(
        hidden_dim=config.hidden_dim,
        num_q_heads=config.num_q_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        ffn_dim=config.ffn_dim,
        num_layers=num_layers,
        vocab_size=config.vocab_size,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        max_seq_len=config.max_seq_len,
        chunk_size=config.chunk_size,
    )

    opts = PostOptOptions(
        row2col_pivot=use_pivot,
        cte_merge=True,
        table_fusion=True,
        pivot_width=pivot_width,
        subquery_width=subquery_width,
    ) if use_pivot else None

    runner = ClickHouseRunner(
        config=config, host=host, port=port, user=user, password=password,
        database=database, postopt=opts,
        max_memory_usage=max_memory_usage, max_threads=max_threads,
    )
    runner.init()
    pivot_setup_time = runner.pivot_setup_time_s

    t0 = time.perf_counter()
    runner.run_prefill(token_ids)

    out_table = runner.get_output_table()

    # Cross-entropy in SQL. arrayJoin(arrayZip(arrayEnumerate(v), v)) keeps
    # position and value aligned (D12: no paired unnest). arrayEnumerate is
    # 1-indexed, hence ``.1 - 1`` to get 0-indexed offsets within the chunk.
    row = runner.client.query(f"""
        WITH
          logits_scalar AS (
            SELECT row_index,
                   chunk_index + _ep.1 - 1 AS vocab_idx,
                   _ep.2 AS val
            FROM (
              SELECT row_index, chunk_index,
                     arrayJoin(arrayZip(arrayEnumerate(v), v)) AS _ep
              FROM {out_table}
            )
          ),
          targets AS (
            SELECT pos - 1 AS t, token_id AS target_tok
            FROM input_tokens WHERE pos > 0
          ),
          mx AS (
            SELECT row_index, max(val) AS mv
            FROM logits_scalar GROUP BY row_index
          ),
          lse AS (
            SELECT l.row_index,
                   m.mv + log(sum(exp(l.val - m.mv))) AS lse_val
            FROM logits_scalar l JOIN mx m ON l.row_index = m.row_index
            GROUP BY l.row_index, m.mv
          ),
          tgt AS (
            SELECT l.row_index, l.val AS tgt_logit
            FROM logits_scalar l
            JOIN targets t ON l.row_index = t.t
                          AND l.vocab_idx = t.target_tok
          )
        SELECT -sum(tgt.tgt_logit - lse.lse_val) AS total_loss,
               count() AS n_tokens
        FROM tgt JOIN lse ON tgt.row_index = lse.row_index
    """).result_rows
    dt = time.perf_counter() - t0

    runner.close()

    chunk_loss = float(row[0][0])
    chunk_tokens = int(row[0][1])
    return chunk_idx, chunk_loss, chunk_tokens, dt, pivot_setup_time


def compute_perplexity(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    num_layers: int,
    chunk_size: int,
    max_chunks: int,
    num_workers: int,
    context_len: int = DEFAULT_CONTEXT_LEN,
    max_memory_usage: int | None = None,
    max_threads: int | None = None,
    use_pivot: bool = True,
    pivot_width: int = 0,
    subquery_width: int = 0,
) -> dict:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading WikiText-2 test set...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(row["text"] for row in ds)
    all_tokens = tokenizer.encode(text, add_special_tokens=False)
    print(f"  Total tokens: {len(all_tokens)}")

    n_chunks = min(max_chunks, len(all_tokens) // context_len)
    print(f"  Chunks: {n_chunks}, context_len: {context_len}, "
          f"workers: {num_workers}")

    chunk_args = [
        (host, port, user, password, database,
         all_tokens[i * context_len:(i + 1) * context_len],
         num_layers, chunk_size, i,
         max_memory_usage, max_threads,
         use_pivot, pivot_width, subquery_width)
        for i in range(n_chunks)
    ]

    total_loss = 0.0
    total_tokens = 0
    pivot_setup_times: list[float] = []

    if num_workers == 1:
        for args in chunk_args:
            chunk_idx, chunk_loss, chunk_tokens, dt, piv_t = _run_chunk(args)
            total_loss += chunk_loss
            total_tokens += chunk_tokens
            pivot_setup_times.append(piv_t)
            ppl_so_far = math.exp(total_loss / total_tokens)
            print(f"  Chunk {chunk_idx+1}/{n_chunks}: "
                  f"{dt:.1f}s, running PPL = {ppl_so_far:.4f}")
    else:
        with multiprocessing.Pool(processes=num_workers) as pool:
            for chunk_idx, chunk_loss, chunk_tokens, dt, piv_t in \
                    pool.imap_unordered(_run_chunk, chunk_args):
                total_loss += chunk_loss
                total_tokens += chunk_tokens
                pivot_setup_times.append(piv_t)
                ppl_so_far = math.exp(total_loss / total_tokens)
                print(f"  Chunk {chunk_idx+1}/{n_chunks}: "
                      f"{dt:.1f}s, running PPL = {ppl_so_far:.4f}")

    final_ppl = math.exp(total_loss / total_tokens)
    pivot_mean = (sum(pivot_setup_times) / len(pivot_setup_times)
                  if pivot_setup_times else 0.0)
    print(f"\nFinal perplexity: {final_ppl:.4f} "
          f"(over {total_tokens} tokens, {n_chunks} chunks)")
    if pivot_mean > 0:
        print(f"Mean pivot setup per chunk (D9, not in latency): "
              f"{pivot_mean:.3f}s")
    return {
        "perplexity": final_ppl,
        "total_tokens": total_tokens,
        "num_chunks": n_chunks,
        "context_length": context_len,
        "num_layers": num_layers,
        "pivot_setup_time_s_mean": pivot_mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ch-host", default="localhost")
    parser.add_argument("--ch-port", type=int, default=8123)
    parser.add_argument("--ch-user", default="default")
    parser.add_argument("--ch-password", default="")
    parser.add_argument("--ch-database", default="default")
    parser.add_argument("--output", default="results/clickhouse_ppl.json")
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--max-chunks", type=int, default=64)
    parser.add_argument("--context-len", type=int,
                        default=DEFAULT_CONTEXT_LEN,
                        help=f"Tokens per evaluation chunk "
                             f"(default: {DEFAULT_CONTEXT_LEN}).")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Parallel workers (default: 1; each opens an "
                             "independent ClickHouse session).")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--no-pivot", action="store_true")
    parser.add_argument("--pivot-width", type=int, default=0)
    parser.add_argument("--subquery-width", type=int, default=0)
    # Paper's hardware: AWS c7.2xlarge = 4 cores, 16 GB RAM.
    parser.add_argument("--max-memory-usage", type=parse_bytes, default=None,
                        help="ClickHouse max_memory_usage. Accepts raw "
                             "bytes or a suffix (e.g. '16GB'). Paper: 16GB.")
    parser.add_argument("--max-threads", type=int, default=None,
                        help="ClickHouse max_threads (paper uses 4)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result = compute_perplexity(
        host=args.ch_host, port=args.ch_port,
        user=args.ch_user, password=args.ch_password,
        database=args.ch_database,
        num_layers=args.num_layers, chunk_size=args.chunk_size,
        max_chunks=args.max_chunks, num_workers=args.num_workers,
        context_len=args.context_len,
        max_memory_usage=args.max_memory_usage,
        max_threads=args.max_threads,
        use_pivot=not args.no_pivot,
        pivot_width=args.pivot_width,
        subquery_width=args.subquery_width,
    )
    result["max_memory_usage"] = args.max_memory_usage
    result["max_threads"] = args.max_threads
    result["use_pivot"] = not args.no_pivot
    result["pivot_width"] = args.pivot_width
    result["subquery_width"] = args.subquery_width
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
