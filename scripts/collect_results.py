"""
collect_results.py — Aggregate TranSQL+ (DuckDB + ClickHouse) and
llama.cpp / DeepSpeed evaluation results.

Reads (in --results-dir):
  - prefill.json                TranSQL+ (DuckDB) prefill (warm)
  - decode.json                 TranSQL+ (DuckDB) decode (warm)
  - transql_ppl.json            TranSQL+ (DuckDB) perplexity
  - clickhouse_prefill.json     TranSQL+ (ClickHouse) prefill (warm)
  - clickhouse_decode.json      TranSQL+ (ClickHouse) decode (warm)
  - clickhouse_ppl.json         TranSQL+ (ClickHouse) perplexity
  - llamacpp_f32_cold.json      llama-bench -o json, cold start
  - llamacpp_f32_warm.json      llama-bench -o json, warm
  - llamacpp_f32_ppl.txt        llama-perplexity stdout
  - deepspeed_bf16_cold.json     DeepSpeed cold (cache dropped externally)
  - deepspeed_bf16_warm.json     DeepSpeed warm

Writes:
  - combined_results.json       one row per (system, run_type, prompt_length)

Usage:
    python scripts/collect_results.py --results-dir results
"""

from __future__ import annotations

import argparse
import json
import os
import re


# ---------------------------------------------------------------------------
# Common entry schema
# ---------------------------------------------------------------------------

def _entry(*, system, run_type, prompt_length,
           prefill_latency_s=None, prefill_latency_std_s=None,
           prefill_throughput_tok_per_s=None,
           decode_latency_s=None, decode_latency_std_s=None,
           decode_throughput_tok_per_s=None,
           peak_rss_mb=None, db_size_gb=None):
    return {
        "system": system,
        "run_type": run_type,
        "prompt_length": prompt_length,
        "prefill_latency_s": prefill_latency_s,
        "prefill_latency_std_s": prefill_latency_std_s,
        "prefill_throughput_tok_per_s": prefill_throughput_tok_per_s,
        "decode_latency_s": decode_latency_s,
        "decode_latency_std_s": decode_latency_std_s,
        "decode_throughput_tok_per_s": decode_throughput_tok_per_s,
        "peak_rss_mb": peak_rss_mb,
        "db_size_gb": db_size_gb,
    }


# ---------------------------------------------------------------------------
# TranSQL+ loaders
# ---------------------------------------------------------------------------

def load_transql(results_dir):
    """TranSQL+ always runs warm (2 discarded + 3 measured, same process)."""
    entries = []
    prefill_path = os.path.join(results_dir, "prefill.json")
    decode_path = os.path.join(results_dir, "decode.json")

    prefill_by_len = {}
    db_size_gb = None
    if os.path.exists(prefill_path):
        with open(prefill_path) as f:
            pdata = json.load(f)
        db_size_gb = pdata.get("db_size_gb")
        for r in pdata.get("results", []):
            prefill_by_len[r["prompt_length"]] = r

    decode_by_len = {}
    if os.path.exists(decode_path):
        with open(decode_path) as f:
            ddata = json.load(f)
        if db_size_gb is None:
            db_size_gb = ddata.get("db_size_gb")
        for r in ddata.get("results", []):
            decode_by_len[r["prompt_length"]] = r

    for length in sorted(set(prefill_by_len) | set(decode_by_len)):
        p = prefill_by_len.get(length, {})
        d = decode_by_len.get(length, {})
        entries.append(_entry(
            system="transql+",
            run_type="warm",
            prompt_length=length,
            prefill_latency_s=p.get("prefill_latency_mean_s"),
            prefill_latency_std_s=p.get("prefill_latency_std_s"),
            prefill_throughput_tok_per_s=p.get("prefill_throughput_tok_per_s"),
            decode_latency_s=d.get("decode_latency_mean_s"),
            decode_latency_std_s=d.get("decode_latency_std_s"),
            decode_throughput_tok_per_s=d.get("decode_throughput_tok_per_s"),
            peak_rss_mb=max(p.get("peak_rss_mb") or 0,
                            d.get("peak_rss_mb") or 0) or None,
            db_size_gb=db_size_gb,
        ))
    return entries


def load_clickhouse(results_dir):
    """TranSQL+ on ClickHouse. Same file layout as load_transql but under
    ``clickhouse_*.json`` filenames and tagged ``system='transql+/ch'``
    so the two backends sit side-by-side in the comparison table."""
    entries = []
    prefill_path = os.path.join(results_dir, "clickhouse_prefill.json")
    decode_path = os.path.join(results_dir, "clickhouse_decode.json")

    prefill_by_len = {}
    db_size_gb = None
    if os.path.exists(prefill_path):
        with open(prefill_path) as f:
            pdata = json.load(f)
        db_size_gb = pdata.get("db_size_gb")
        for r in pdata.get("results", []):
            prefill_by_len[r["prompt_length"]] = r

    decode_by_len = {}
    if os.path.exists(decode_path):
        with open(decode_path) as f:
            ddata = json.load(f)
        for r in ddata.get("results", []):
            decode_by_len[r["prompt_length"]] = r

    for length in sorted(set(prefill_by_len) | set(decode_by_len)):
        p = prefill_by_len.get(length, {})
        d = decode_by_len.get(length, {})
        entries.append(_entry(
            system="transql+/ch",
            run_type="warm",
            prompt_length=length,
            prefill_latency_s=p.get("prefill_latency_mean_s"),
            prefill_latency_std_s=p.get("prefill_latency_std_s"),
            prefill_throughput_tok_per_s=p.get("prefill_throughput_tok_per_s"),
            decode_latency_s=d.get("decode_latency_mean_s"),
            decode_latency_std_s=d.get("decode_latency_std_s"),
            decode_throughput_tok_per_s=d.get("decode_throughput_tok_per_s"),
            peak_rss_mb=max(p.get("peak_rss_mb") or 0,
                            d.get("peak_rss_mb") or 0) or None,
            db_size_gb=db_size_gb,
        ))
    return entries


def load_deepspeed(results_dir, variant="bf16"):
    """DeepSpeed baseline. One JSON per run_type, each carrying a list of
    per-length results with both prefill and decode stats."""
    entries = []
    for run_type in ("cold", "warm"):
        path = os.path.join(results_dir,
                            f"deepspeed_{variant}_{run_type}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for r in data.get("results", []):
            entries.append(_entry(
                system=f"deepspeed_{variant}",
                run_type=run_type,
                prompt_length=r.get("prompt_length"),
                prefill_latency_s=r.get("prefill_latency_mean_s"),
                prefill_latency_std_s=r.get("prefill_latency_std_s"),
                prefill_throughput_tok_per_s=r.get(
                    "prefill_throughput_tok_per_s"),
                decode_latency_s=r.get("decode_latency_mean_s"),
                decode_latency_std_s=r.get("decode_latency_std_s"),
                decode_throughput_tok_per_s=r.get(
                    "decode_throughput_tok_per_s"),
                peak_rss_mb=r.get("peak_rss_mb"),
            ))
    return entries


# ---------------------------------------------------------------------------
# llama.cpp loaders
# ---------------------------------------------------------------------------

def _parse_llama_bench_json(path):
    """Return list of test dicts from llama-bench -o json output.

    llama-bench -o json emits either a list of test objects or a single dict
    (older builds). We normalise to list.
    """
    if not os.path.exists(path):
        return []
    with open(path) as f:
        text = f.read().strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: extract the outermost JSON array if there is trailing text.
        m = re.search(r"\[[\s\S]*\]", text)
        if not m:
            return []
        data = json.loads(m.group(0))
    return data if isinstance(data, list) else [data]


def load_llamacpp(results_dir, variant="f32"):
    """Build one row per (system, run_type, prompt_length) for llama.cpp.

    llama-bench `-p A,B -n C` produces:
      * pp tests:  n_prompt in {A,B}, n_gen == 0
      * tg tests:  n_prompt == 0,     n_gen == C  (decode from empty context)

    The tg test is prompt-length-independent in llama-bench's model, so we
    broadcast its single throughput value to every pp prompt length.
    """
    entries = []

    for run_type, fname in [
        ("cold", f"llamacpp_{variant}_cold.json"),
        ("warm", f"llamacpp_{variant}_warm.json"),
    ]:
        tests = _parse_llama_bench_json(os.path.join(results_dir, fname))
        if not tests:
            continue

        pp_tests = {int(t["n_prompt"]): t for t in tests
                    if t.get("n_gen") == 0 and t.get("n_prompt")}
        tg_tests = [t for t in tests
                    if t.get("n_prompt") == 0 and t.get("n_gen", 0) > 0]
        tg = tg_tests[0] if tg_tests else None

        tg_ts = tg.get("avg_ts") if tg else None
        tg_std_ts = tg.get("stddev_ts") if tg else None
        decode_s = (1.0 / tg_ts) if tg_ts else None
        # 1/x Taylor: std(1/x) ≈ std(x) / x^2
        decode_std_s = (tg_std_ts / (tg_ts ** 2)) if (tg_std_ts and tg_ts) else None

        for length in sorted(pp_tests):
            pp = pp_tests[length]
            avg_ns = pp.get("avg_ns")
            stddev_ns = pp.get("stddev_ns")
            avg_ts = pp.get("avg_ts")

            prefill_s = (avg_ns / 1e9) if avg_ns else (
                length / avg_ts if avg_ts else None)
            prefill_std_s = (stddev_ns / 1e9) if stddev_ns else None

            entries.append(_entry(
                system=f"llamacpp_{variant}",
                run_type=run_type,
                prompt_length=length,
                prefill_latency_s=prefill_s,
                prefill_latency_std_s=prefill_std_s,
                prefill_throughput_tok_per_s=avg_ts,
                decode_latency_s=decode_s,
                decode_latency_std_s=decode_std_s,
                decode_throughput_tok_per_s=tg_ts,
                peak_rss_mb=None,  # llama-bench JSON does not report RSS
            ))

    return entries


# ---------------------------------------------------------------------------
# Perplexity loaders
# ---------------------------------------------------------------------------

def _parse_llama_perplexity_text(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    m = re.search(r"[Ff]inal.*?PPL\s*=\s*(\d+\.\d+)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"perplexity\s*[:=]\s*(\d+\.\d+)", text)
    return float(m.group(1)) if m else None


def load_perplexity(results_dir):
    ppl = {}
    transql_path = os.path.join(results_dir, "transql_ppl.json")
    if os.path.exists(transql_path):
        with open(transql_path) as f:
            ppl["transql+"] = json.load(f)

    ch_ppl = os.path.join(results_dir, "clickhouse_ppl.json")
    if os.path.exists(ch_ppl):
        with open(ch_ppl) as f:
            ppl["transql+/ch"] = json.load(f)

    for variant in ["f32"]:
        v = _parse_llama_perplexity_text(
            os.path.join(results_dir, f"llamacpp_{variant}_ppl.txt"))
        if v is not None:
            ppl[f"llamacpp_{variant}"] = {"perplexity": v}

    return ppl


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _fmt(v, fmt=".3f"):
    return f"{v:{fmt}}" if v is not None else "--"


def print_table(entries, perplexity):
    sys_keys = sorted({(e["system"], e["run_type"]) for e in entries})
    lengths = sorted({e["prompt_length"] for e in entries})
    labels = [f"{s}/{rt}" for (s, rt) in sys_keys]
    col_w = max(18, max((len(l) for l in labels), default=18) + 2)

    def get(system, run_type, length, field):
        for e in entries:
            if (e["system"] == system and e["run_type"] == run_type
                    and e["prompt_length"] == length):
                return e.get(field)
        return None

    print("\n" + "=" * (10 + col_w * len(sys_keys)))
    print("  TranSQL+ (DuckDB/ClickHouse) vs llama.cpp / DeepSpeed"
          " — prefill / decode comparison")
    print("=" * (10 + col_w * len(sys_keys)))

    header = f"{'Prompt':>10}" + "".join(f"{l:>{col_w}}" for l in labels)

    for title, field in [
        ("Prefill latency (s)    — time to first token",
         "prefill_latency_s"),
        ("Prefill throughput (tok/s)", "prefill_throughput_tok_per_s"),
        ("Decode latency (s/tok) — avg per token after first",
         "decode_latency_s"),
        ("Decode throughput (tok/s)", "decode_throughput_tok_per_s"),
    ]:
        print(f"\n--- {title} ---")
        print(header)
        print("-" * len(header))
        for length in lengths:
            row = f"{length:>10}"
            for (s, rt) in sys_keys:
                row += f"{_fmt(get(s, rt, length, field)):>{col_w}}"
            print(row)

    print(f"\n--- Peak RSS (MB) ---")
    print(f"{'':>10}" + "".join(f"{l:>{col_w}}" for l in labels))
    print("-" * (10 + col_w * len(sys_keys)))
    row = f"{'Peak':>10}"
    for (s, rt) in sys_keys:
        vals = [e["peak_rss_mb"] for e in entries
                if e["system"] == s and e["run_type"] == rt
                and e["peak_rss_mb"]]
        row += f"{_fmt(max(vals), '.0f') if vals else '--':>{col_w}}"
    print(row)

    db_sizes = {e["db_size_gb"] for e in entries if e.get("db_size_gb")}
    if db_sizes:
        print(f"\n  TranSQL+ DuckDB size: {max(db_sizes):.2f} GB")

    if perplexity:
        print(f"\n--- Perplexity (WikiText-2, lower = better) ---")
        for name in sorted(perplexity):
            blob = perplexity[name]
            v = blob.get("perplexity") if isinstance(blob, dict) else blob
            print(f"  {name:>20}: {_fmt(v, '.4f')}")

    print("\n" + "=" * (10 + col_w * len(sys_keys)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default=None,
                        help="Combined JSON path "
                             "(default: <results-dir>/combined_results.json)")
    args = parser.parse_args()

    transql_entries = load_transql(args.results_dir)
    clickhouse_entries = load_clickhouse(args.results_dir)
    llamacpp_entries = load_llamacpp(args.results_dir, variant="f32")
    deepspeed_entries = load_deepspeed(args.results_dir, variant="bf16")
    all_entries = (transql_entries + clickhouse_entries
                   + llamacpp_entries + deepspeed_entries)
    perplexity = load_perplexity(args.results_dir)

    print_table(all_entries, perplexity)

    out_path = args.output or os.path.join(
        args.results_dir, "combined_results.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "results": all_entries,
            "perplexity": perplexity,
        }, f, indent=2, default=str)
    print(f"\nCombined results saved to {out_path}")


if __name__ == "__main__":
    main()
