"""
DeepSpeed baseline for the TranSQL+ paper comparison.

Loads Llama-3-8B via HuggingFace ``transformers`` and wraps it with
``deepspeed.init_inference`` (v1 API, CPU BF16). Reports prefill latency
(time to first token) and decode latency (token-by-token with
``past_key_values``) for prompt lengths {25, 50, 100, 200}, using the
same measurement protocol as ``scripts/run_prefill.py`` and
``scripts/run_decode.py`` so ``scripts/collect_results.py`` can load
the output with a shared schema.

Protocol:
  * Load model + DeepSpeed.init_inference (once).
  * Per prompt length L (read ``prompts/prompt_{L}.json``):
      - 2 warmup prefill runs (discarded).
      - 3 measured prefill runs.
      - 2 warmup decode steps (discarded).
      - 49 measured decode steps (reusing the prefill KV cache).

Usage:
    # Warm (no cache drop)
    python scripts/run_deepspeed.py \\
        --output results/deepspeed_bf16_warm.json

    # Cold (the user drops caches beforehand via
    # ``sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'``)
    python scripts/run_deepspeed.py --cold \\
        --output results/deepspeed_bf16_cold.json

Matches the paper's c7.2xlarge profile: ``--threads 4`` sets
``torch.set_num_threads`` and ``OMP_NUM_THREADS`` for parity with
llama.cpp's ``-t 4``.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time

# IMPORTANT: thread-count and SIMD env vars must be set before torch imports.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
# Cap MKL to SSE4.2 code paths (no AVX/AVX2) for SIMD-matched comparison
# with the paper's AWS c7g.2xlarge (ARM NEON, 128-bit).  Set
# MKL_CBWR=AVX2 on the command line to restore full AVX2.
os.environ.setdefault("MKL_CBWR", "AVX")

import numpy as np                       # noqa: E402
import torch                             # noqa: E402


MODEL_ID = "meta-llama/Meta-Llama-3-8B"
WARMUP_PREFILL = 2
MEASURED_PREFILL = 3
WARMUP_DECODE = 2
MEASURED_DECODE = 49


def get_peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_model(threads: int):
    """HF model + DeepSpeed init_inference wrap (v1 API, CPU BF16)."""
    import deepspeed
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(threads)
    print(f"Loading {MODEL_ID} in bfloat16 (threads={threads})...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
    )
    model.eval()
    print(f"  HF load: {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    ds_engine = deepspeed.init_inference(
        model,
        dtype=torch.bfloat16,
        mp_size=1,
        replace_with_kernel_inject=False,
    )
    print(f"  DeepSpeed init_inference: "
          f"{time.perf_counter() - t0:.1f}s")
    return ds_engine


@torch.inference_mode()
def run_prefill(engine, input_ids: torch.Tensor) -> tuple[float, object, torch.Tensor]:
    """One prefill pass; returns (latency_s, past_key_values, next_token)."""
    t0 = time.perf_counter()
    out = engine(input_ids=input_ids, use_cache=True)
    dt = time.perf_counter() - t0
    # Greedy argmax on the last-position logits.
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return dt, out.past_key_values, next_token


@torch.inference_mode()
def run_decode_step(engine, token: torch.Tensor,
                    past_kv) -> tuple[float, object, torch.Tensor]:
    """One decode step; returns (latency_s, updated_kv, next_token)."""
    t0 = time.perf_counter()
    out = engine(input_ids=token, past_key_values=past_kv, use_cache=True)
    dt = time.perf_counter() - t0
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return dt, out.past_key_values, next_token


def measure_one_length(engine, prompt_path: str, *,
                       prefill_warmup: int, prefill_measured: int,
                       decode_warmup: int, decode_measured: int) -> dict:
    with open(prompt_path) as f:
        prompt = json.load(f)
    token_ids = prompt["token_ids"]
    seq_len = len(token_ids)
    input_ids = torch.tensor([token_ids], dtype=torch.long)

    # -------- Prefill --------
    print(f"  Prefill warmup: {prefill_warmup}")
    for w in range(prefill_warmup):
        dt, _, _ = run_prefill(engine, input_ids)
        print(f"    warmup {w+1}: {dt:.3f}s (discarded)")

    prefill_latencies: list[float] = []
    last_kv = None
    last_next = None
    for r in range(prefill_measured):
        dt, kv, nxt = run_prefill(engine, input_ids)
        prefill_latencies.append(dt)
        last_kv = kv
        last_next = nxt
        print(f"    run {r+1}: {dt:.3f}s ({seq_len/dt:.2f} tok/s)")

    # -------- Decode (reuse the KV cache from the last prefill) --------
    total_decode_steps = decode_warmup + decode_measured
    all_decode: list[float] = []
    kv = last_kv
    tok = last_next
    for step in range(total_decode_steps):
        dt, kv, tok = run_decode_step(engine, tok, kv)
        all_decode.append(dt)
        phase = "warmup" if step < decode_warmup else "measure"
        if (step < decode_warmup
                or (step - decode_warmup) % 10 == 0
                or step == total_decode_steps - 1):
            print(f"    [{phase}] step {step+1}/{total_decode_steps}: "
                  f"{dt:.3f}s ({1.0/dt:.2f} tok/s)")

    measured_decode = all_decode[decode_warmup:]

    mean_pref = float(np.mean(prefill_latencies))
    std_pref = float(np.std(prefill_latencies))
    mean_dec = float(np.mean(measured_decode))
    std_dec = float(np.std(measured_decode))

    return {
        "prompt_length": seq_len,
        "prefill_latencies_s": prefill_latencies,
        "prefill_latency_mean_s": mean_pref,
        "prefill_latency_std_s": std_pref,
        "prefill_throughput_tok_per_s": seq_len / mean_pref,
        "decode_latencies_s": measured_decode,
        "decode_latency_mean_s": mean_dec,
        "decode_latency_std_s": std_dec,
        "decode_throughput_tok_per_s": 1.0 / mean_dec,
        "warmup_runs": prefill_warmup,
        "measured_runs": prefill_measured,
        "warmup_steps": decode_warmup,
        "measured_steps": decode_measured,
        "peak_rss_mb": get_peak_rss_mb(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-dir", default="prompts")
    parser.add_argument("--output", default="results/deepspeed_bf16_warm.json")
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[25, 50, 100, 200])
    parser.add_argument("--prefill-warmup", type=int, default=WARMUP_PREFILL)
    parser.add_argument("--prefill-repeat", type=int, default=MEASURED_PREFILL)
    parser.add_argument("--decode-warmup", type=int, default=WARMUP_DECODE)
    parser.add_argument("--decode-steps", type=int, default=MEASURED_DECODE)
    parser.add_argument("--threads", type=int, default=4,
                        help="torch.set_num_threads + OMP_NUM_THREADS "
                             "(paper: 4).")
    parser.add_argument("--cold", action="store_true",
                        help="Label this run as cold (user drops the page "
                             "cache before invocation). Does not itself "
                             "drop caches.")
    args = parser.parse_args()

    run_type = "cold" if args.cold else "warm"
    print(f"DeepSpeed baseline ({run_type} run, threads={args.threads})")
    print(f"Protocol: prefill {args.prefill_warmup}w+{args.prefill_repeat}m, "
          f"decode {args.decode_warmup}w+{args.decode_steps}m")
    print()

    engine = load_model(args.threads)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results = []

    for length in args.lengths:
        prompt_path = os.path.join(args.prompts_dir, f"prompt_{length}.json")
        if not os.path.exists(prompt_path):
            print(f"SKIP: {prompt_path} not found")
            continue

        print(f"Prompt length {length}:")
        r = measure_one_length(
            engine, prompt_path,
            prefill_warmup=args.prefill_warmup,
            prefill_measured=args.prefill_repeat,
            decode_warmup=args.decode_warmup,
            decode_measured=args.decode_steps,
        )
        results.append(r)
        print(f"  => Prefill: {r['prefill_latency_mean_s']:.3f}s "
              f"(+/- {r['prefill_latency_std_s']:.3f}s)  "
              f"{r['prefill_throughput_tok_per_s']:.2f} tok/s")
        print(f"  => Decode:  {r['decode_latency_mean_s']:.3f}s/tok "
              f"(+/- {r['decode_latency_std_s']:.3f}s)  "
              f"{r['decode_throughput_tok_per_s']:.2f} tok/s  "
              f"RSS: {r['peak_rss_mb']:.0f} MB\n")

    output = {
        "system": "deepspeed_bf16",
        "run_type": run_type,
        "model_id": MODEL_ID,
        "threads": args.threads,
        "prefill_warmup": args.prefill_warmup,
        "prefill_measured": args.prefill_repeat,
        "decode_warmup": args.decode_warmup,
        "decode_measured": args.decode_steps,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
