"""
Measure TranSQL+ perplexity on WikiText-2.

Paper Section 5: perplexity evaluation against llama.cpp.

Computes perplexity by running prefill on WikiText-2 chunks, computing
cross-entropy loss entirely in SQL (avoids fetching seq_len x vocab_size
rows to Python). Uses logsumexp CTE for numerical stability.

Usage:
    python scripts/run_perplexity.py \
        --db-path weights.duckdb \
        --output results/transql_ppl.json \
        [--num-layers 32] [--max-chunks 64] [--num-workers 1]

Requires: pip install datasets transformers
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import time

from transql_plus.config import ModelConfig
from transql_plus.postopt import PostOptOptions
from transql_plus.runner import TranSQLRunner


MODEL_ID = "meta-llama/Meta-Llama-3-8B"
# Default chunk size inherited from the original implementation. The
# TranSQL+ paper does not prescribe a specific WikiText-2 evaluation
# chunk length that we have been able to locate; 512 is a conventional
# value in LLM perplexity evaluation (e.g. llama.cpp's default). Set
# --context-len to override for smoke tests or alternative baselines.
DEFAULT_CONTEXT_LEN = 512


def _run_chunk(args: tuple) -> tuple[int, float, int, float, float]:
    """Worker: run prefill on one chunk and compute cross-entropy in SQL.

    Cross-entropy is computed entirely inside DuckDB to avoid transferring
    the full logits matrix (seq_len x vocab_size) to Python.

    SQL approach: logsumexp for numerical stability:
      loss = -SUM(logit[target] - logsumexp(logits_row))

    Returns (chunk_idx, chunk_loss, chunk_tokens, dt, pivot_setup_time_s).
    """
    (db_path, token_ids, num_layers, chunk_size, chunk_idx,
     memory_limit, threads, temp_directory,
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

    runner = TranSQLRunner(
        db_path, config, postopt=opts, read_only=True,
        memory_limit=memory_limit,
        threads=threads,
        temp_directory=temp_directory,
    )
    runner.init()
    pivot_setup_time = runner.pivot_setup_time_s

    t0 = time.perf_counter()
    runner.run_prefill(token_ids)

    # Compute cross-entropy in SQL.
    # The logits are in chunked format: (row_index, chunk_index, v FLOAT[cs]).
    # First unnest to scalar format, then compute logsumexp + target logit.
    out_table = runner.get_output_table()

    row = runner.con.execute(f"""
        WITH
          logits_scalar AS (
            SELECT row_index,
                   chunk_index + UNNEST(generate_series(0, len(v)-1)) AS vocab_idx,
                   UNNEST(v) AS val
            FROM {out_table}
          ),
          targets AS (
            SELECT pos - 1 AS t, token_id AS target_tok
            FROM input_tokens WHERE pos > 0
          ),
          mx AS (
            SELECT row_index, MAX(val) AS mv
            FROM logits_scalar GROUP BY row_index
          ),
          lse AS (
            SELECT l.row_index,
                   m.mv + LOG(SUM(EXP(l.val - m.mv))) AS lse_val
            FROM logits_scalar l JOIN mx m ON l.row_index = m.row_index
            GROUP BY l.row_index, m.mv
          ),
          tgt AS (
            SELECT l.row_index, l.val AS tgt_logit
            FROM logits_scalar l
            JOIN targets t ON l.row_index = t.t
                          AND l.vocab_idx = t.target_tok
          )
        SELECT -SUM(tgt.tgt_logit - lse.lse_val) AS total_loss,
               COUNT(*) AS n_tokens
        FROM tgt JOIN lse ON tgt.row_index = lse.row_index
    """).fetchone()
    dt = time.perf_counter() - t0

    runner.close()

    chunk_loss = float(row[0])
    chunk_tokens = int(row[1])
    return chunk_idx, chunk_loss, chunk_tokens, dt, pivot_setup_time


def compute_perplexity(
    db_path: str,
    num_layers: int,
    chunk_size: int,
    max_chunks: int,
    num_workers: int,
    *,
    context_len: int = DEFAULT_CONTEXT_LEN,
    memory_limit: str | None = None,
    threads: int | None = None,
    temp_directory: str | None = None,
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
        (db_path,
         all_tokens[i * context_len:(i + 1) * context_len],
         num_layers, chunk_size, i,
         memory_limit, threads, temp_directory,
         use_pivot, pivot_width, subquery_width)
        for i in range(n_chunks)
    ]

    total_loss = 0.0
    total_tokens = 0
    pivot_setup_times: list[float] = []

    if num_workers == 1:
        # Single-process: no multiprocessing overhead
        for args in chunk_args:
            chunk_idx, chunk_loss, chunk_tokens, dt, piv_t = \
                _run_chunk(args)
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
    parser = argparse.ArgumentParser(
        description="Measure TranSQL+ perplexity on WikiText-2"
    )
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--output", default="results/transql_ppl.json")
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--max-chunks", type=int, default=64)
    parser.add_argument("--context-len", type=int,
                        default=DEFAULT_CONTEXT_LEN,
                        help=f"Tokens per evaluation chunk "
                             f"(default: {DEFAULT_CONTEXT_LEN}). Use a "
                             "small value (e.g. 25) for smoke tests to "
                             "match prefill/decode seq_len.")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Parallel workers (default: 1, memory-safe)")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--no-pivot", action="store_true",
                        help="Disable ROW2COL pivot (baseline comparison)")
    parser.add_argument("--pivot-width", type=int, default=0,
                        help="ROW2COL pivot_width (§4.3). 0 = single PIVOT "
                             "over all chunks (default; required for weight "
                             "pivot caching in runner.py). Tune via "
                             "scripts/tune_pivot.py.")
    parser.add_argument("--subquery-width", type=int, default=0,
                        help="ROW2COL subquery_width (§4.3). 0 = 1 dot-product "
                             "per CROSS JOIN CTE (paper example pattern). "
                             "Tune via scripts/tune_pivot.py.")
    # Paper's hardware: AWS c7.2xlarge = 4 cores, 16 GB RAM.
    parser.add_argument("--memory-limit", default=None,
                        help="DuckDB memory_limit per worker (e.g. '16GB'). "
                             "Paper uses 16GB RAM — set matching value here.")
    parser.add_argument("--threads", type=int, default=None,
                        help="DuckDB threads per worker (paper uses 4)")
    parser.add_argument("--temp-directory", default=None,
                        help="DuckDB temp_directory for spill files")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result = compute_perplexity(
        args.db_path, args.num_layers, args.chunk_size,
        args.max_chunks, args.num_workers,
        context_len=args.context_len,
        memory_limit=args.memory_limit,
        threads=args.threads,
        temp_directory=args.temp_directory,
        use_pivot=not args.no_pivot,
        pivot_width=args.pivot_width,
        subquery_width=args.subquery_width,
    )
    result["memory_limit"] = args.memory_limit
    result["threads"] = args.threads
    result["temp_directory"] = args.temp_directory
    result["use_pivot"] = not args.no_pivot
    result["pivot_width"] = args.pivot_width
    result["subquery_width"] = args.subquery_width
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
