# TranSQL+ Reproduction (arxiv 2502.02818)

Independent reproduction of the TranSQL+ paper: running LLM inference entirely inside a relational database. Primary backend is DuckDB; Step 4B ports the same SQL pipeline to ClickHouse (paper-portability claim). Step 5 adds two native-inference baselines (llama.cpp, DeepSpeed).

## Setup

```bash
conda create -n llm_db python=3
conda activate llm_db
pip install -r requirements.txt
pip3 install -e .

# Login to Hugging Face (needed to download Llama3-8B weights)
hf auth login
```

### Download Llama3-8B

Accept the license at https://huggingface.co/meta-llama/Meta-Llama-3-8B, then:

```bash
hf download meta-llama/Meta-Llama-3-8B
```

This caches the model to `~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/<hash>/`
(e.g., `~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/`).
The `extract_weights.py` script loads it by model ID automatically, so you do not need to know the exact path.

### Download WikiText-2 (for perplexity)

For `run_perplexity.py` (uses HuggingFace `datasets` library — downloads automatically, no manual step needed).

For llama.cpp's `llama-perplexity` (needs the raw text file):

```bash
wget https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip
```

This produces `wikitext-2-raw/wiki.test.raw`.

## Unit Tests

```bash
python -m pytest tests/ -v
```

Individual test files:

```bash
python -m pytest tests/test_sql_templates.py -v    # SQL operator templates (Section 3.2)
python -m pytest tests/test_postopt.py -v           # Post-optimisations (Section 4)
python -m pytest tests/test_single_layer.py -v      # Full transformer layer
python -m pytest tests/test_preprocessing.py -v     # Weight preprocessing (Section 3.1)
python -m pytest tests/test_runner.py -v             # Inference runner (Section 5)
```

## Step 1: Extract and Preprocess Weights

```bash
# Extract weights from HuggingFace (downloads ~15GB model)
python preprocessing/extract_weights.py \
    --output-dir weights_npy \
    --source pytorch

# Convert .npy to chunked CSV (Algorithm 1)
python preprocessing/preprocess_weights.py \
    --npy-dir weights_npy \
    --csv-dir weights_csv \
    --chunk-size 32

# Load CSV into DuckDB
python preprocessing/load_weights_duckdb.py \
    --csv-dir weights_csv \
    --db-path weights.duckdb \
    --chunk-size 32
```

Optional: copy to RAM for faster I/O:

```bash
sudo mount -o remount,size=40G /dev/shm
cp weights.duckdb /dev/shm/weights.duckdb
# Then use --db-path /dev/shm/weights.duckdb in all commands below
```

## Step 2: Sample Prompts

```bash
python scripts/sample_prompts.py \
    --output-dir prompts \
    --lengths 25 50 100 200
```

This downloads LMSys-Chat-1M and selects prompts closest to each target length. Outputs `prompts/prompt_25.json`, etc.

### (Optional) Tune ROW2COL pivot config — once per machine

Paper §4.3 notes that `(pivot_width, subquery_width)` is hardware-dependent: "too many projections in one query can exceed parallelism, while too many subqueries increase I/O." Before running the benchmarks, find the best pair for your machine:

```bash
mkdir -p duckdb_tmp
python scripts/tune_pivot.py \
    --chunk-size 32 --threads 4 \
    --lengths 25,50,100,200 \
    --memory-limit 16GB --temp-directory ./duckdb_tmp
```

This runs synthetic MatMuls at all four Llama3-8B projection shapes (Q/O, K/V, gate/up, down) × every prompt length, matching what appears in Llama3-8B prefill. It prints per-matrix bests plus a **global recommendation** (the `(pivot_width, subquery_width)` pair with the best geometric-mean speedup across all tested matrices). Pass that pair to every benchmark via `--pivot-width X --subquery-width Y`.

**Alignment with the full benchmark.** The script mirrors `run_prefill.py` so the tuned pair transfers correctly:
- **`--memory-limit` / `--temp-directory` / `--threads`** — pass the same values you'll use for the full benchmark (e.g. paper's 16 GB / 4 cores). Running tuning with different DuckDB constraints than measurement produces a tuned pair that may not be optimal under the real constraints.
- **CTE merge (§4.1)** — pivoted sub-steps are wrapped in one `WITH … AS` block per run, like `postopt_dag_to_sql` emits when `cte_merge=True`.
- **D9 weight-pivot cache** — weight pivots are materialized once per `pivot_width` outside the timed loop (mirroring `runner.py`'s cache).
- **`should_pivot` heuristic** — matrices with `n_chunks > 128` (default `max_pivot_chunks`) are printed as `SKIP`, because the real runner never pivots them. For `chunk_size=32` the `down` projection (`n_chunks=448`) hits this; for `chunk_size=64` it still hits (`n_chunks=224`).

`chunk_size` must match your stored `weights.duckdb` (default 32 for the reproduction). To compare 32 vs 64 theoretically, re-run with `--chunk-size 64` — but actually using `chunk_size=64` requires rebuilding `weights.duckdb` from scratch via `preprocessing/preprocess_weights.py`.

#### ClickHouse variant — only `pivot_width` is tunable

On ClickHouse there is no `POSITIONAL JOIN`, so the §4.3 rewrite collapses the whole per-chunk dot-product into a single `SELECT` (see `transql_plus/clickhouse/postopt_ch.py::pivoted_matmul_sql`). `subquery_width` becomes a no-op — accepted for CLI parity but ignored. Only `pivot_width` needs tuning:

```bash
python scripts/tune_pivot_clickhouse.py \
    --ch-host localhost --ch-port 8123 --ch-database default \
    --chunk-size 32 --max-threads 4 \
    --lengths 25,50,100,200 \
    --max-memory-usage 16GB
```

Mirrors `run_clickhouse_prefill.py`: same D9 weight-pivot cache, §4.1 CTE merge, `should_pivot` heuristic, and session settings. Prints per-matrix bests plus a global recommendation; pass it to every ClickHouse benchmark via `--pivot-width X` (drop `--subquery-width` entirely — it is ignored).

**Weight pivot caching** (Decision D9): as of the current revision, `runner.py` caches pivoted weights as TEMP TABLEs across inference runs for **any** `pivot_width` value (including multi-group `_piv_g{N}` schemas). The one-time pivot cost is reported as `pivot_setup_time_s` in the output JSON and printed at the start of each measurement — it is not counted in the per-run latency. See `scripts/bench_pivot_cache.py` for the benchmark that justifies this (192× per-run speedup, break-even at ~1.2 runs).

### Note — decode × QKV fusion fix (D11)

`run_decode.py` fails with `Catalog Error: Table with name l0_v does not exist!` when QKV fusion (§4.2) is active. Fusion absorbs the per-layer `l{l}_v` output into a single `l{l}_q_qkv` table with a `flag` column; decode's KV-cache append `INSERT INTO l{l}_v ...` has no target to write into. Neither the paper nor AQP_middleware documents this interaction (see `reproduction_note.md` D11).

**Fix applied** — decision D11 in `transql_plus/runner.py`: keep fusion ON (paper-faithful) and, on the first decode step after prefill, extract `l{l}_v` once from `l{l}_q_qkv WHERE flag='V'`. One CTAS per layer of `seq_len × kv_chunks` rows, run before the decode timing window opens so it does not contaminate `decode_latency_mean_s`. The standard 2-warmup-step protocol (`reproduction_note.md` §Measurement Protocol) also absorbs any first-step cost. No changes are required in `run_decode.py`, `run_prefill.py`, or `run_perplexity.py`.

The alternative — disabling fusion entirely — was rejected because it deviates from paper §4.2. D11 is the minimal fix: §4.2 semantics preserved, §5's silent implementation gap patched.

#### (Optional diagnostic) Measure how §4.2 fusion actually performs on your hardware

`scripts/bench_d11_overhead.py` is provided for readers who want to investigate fusion's real impact. It runs variant **A** (fusion ON + D11) vs **B** (fusion OFF) under the real-benchmark DuckDB constraints, times each phase separately, and reports `Fusion benefit at prefill` (positive = §4.2 delivers as claimed; negative = fusion costs more than it saves on this config). This does **not** influence the reproduction's choice — D11 stays in place regardless — but the result is informative for `reproduction_note.md` commentary on paper-vs-reality.

A 1-layer probe (`--num-layers 1 --lengths 25`) found Option A's D11 materialize was 1.72 ms (0.015% of prefill) — confirming the D11 overhead itself is negligible — while A's prefill was 2.6s slower than B's. That 1-layer result is not conclusive for 32 layers; for a definitive datum at real scale, run:

```bash
python scripts/bench_d11_overhead.py \
    --db-path weights.duckdb \
    --num-layers 32 \
    --lengths 25 \
    --pivot-width 32 --subquery-width 4 \
    --memory-limit 16GB --threads 4 \
    --temp-directory ./duckdb_tmp \
    --warmup 1 --measure 3
```

Expected runtime ~15-20 min. Interpret the result only as paper-vs-reality commentary; the fix is already locked in.

## Step 3: Smoke Tests (1 layer, fast)

Smoke tests run once without warmup — just enough to verify the pipeline end-to-end.

Two flags are required even in smoke tests:
- `--temp-directory ./duckdb_tmp` — without it, DuckDB spills to `weights.duckdb.tmp/` next to the 35 GB DB and can fill that path with PIVOT / CROSS JOIN intermediates.
- `--memory-limit 16GB --threads 4` — matches the paper's `reproduction_note.md` §Measurement Protocol. Without `--memory-limit`, DuckDB defaults to ~80% of host RAM; certain operators (hash aggregation in particular) don't spill gracefully once that ceiling is reached, producing `OutOfMemoryException` at large seq_len.

Note on perplexity: `run_perplexity.py` defaults to `--context-len 512` tokens per chunk (a conventional perplexity chunk size, e.g. llama.cpp's default — the TranSQL+ paper does not prescribe a specific value that we have located). At 1 layer + `lm_head` this can still exhaust tens of GB of RAM. For a smoke test, set `--context-len 25` so it matches the prefill/decode seq_len and completes in seconds.

```bash
mkdir -p duckdb_tmp

# Smoke test prefill (1 layer, 1 run, no warmup)
python scripts/run_prefill.py \
    --db-path weights.duckdb \
    --num-layers 1 \
    --lengths 25 \
    --warmup 0 --repeat 1 \
    --memory-limit 16GB --threads 4 \
    --temp-directory ./duckdb_tmp \
    --pivot-width 32 --subquery-width 4

# Smoke test decode (1 layer, 1 step, no warmup)
python scripts/run_decode.py \
    --db-path weights.duckdb \
    --num-layers 1 \
    --lengths 25 \
    --warmup 0 --decode-steps 1 \
    --memory-limit 16GB --threads 4 \
    --temp-directory ./duckdb_tmp \
    --pivot-width 32 --subquery-width 4

# Smoke test perplexity (1 layer, 1 chunk, seq_len=25 to match the others)
python scripts/run_perplexity.py \
    --db-path weights.duckdb \
    --num-layers 1 \
    --max-chunks 1 \
    --context-len 25 \
    --memory-limit 16GB --threads 4 \
    --temp-directory ./duckdb_tmp \
    --pivot-width 32 --subquery-width 4
```

## Step 4: Full Benchmarks (32 layers)

The paper runs on **AWS c7.2xlarge: 4 physical cores, 16 GB RAM**, explicitly targeting "memory-constrained conditions" with DuckDB's out-of-core execution. To reproduce faithfully on a larger host, constrain DuckDB:

- `--memory-limit 16GB` — matches the paper's 16 GB RAM instance
- `--threads 4` — matches c7.2xlarge physical cores
- `--temp-directory ./duckdb_tmp` — explicit spill location for out-of-core execution

See `reproduction_note.md` → "Hardware Constraint" for rationale.

### Prefill (time to first token)

Protocol: 2 warmup runs (discard) + 3 measured runs per prompt length.

```bash
mkdir -p duckdb_tmp
python scripts/run_prefill.py \
    --db-path weights.duckdb \
    --prompts-dir prompts \
    --output results/prefill.json \
    --pivot-width 32 --subquery-width 4 \
    --lengths 25 50 100 200 \
    --memory-limit 16GB --threads 4 \
    --temp-directory ./duckdb_tmp
```

### Decode (time per output token)

Protocol: prefill first, then 2 warmup decode steps (discard) + 49 measured steps.

```bash
python scripts/run_decode.py \
    --db-path weights.duckdb \
    --prompts-dir prompts \
    --output results/decode.json \
    --pivot-width 32 --subquery-width 4 \
    --lengths 25 50 100 200 \
    --memory-limit 16GB --threads 4 \
    --temp-directory ./duckdb_tmp
```

### Perplexity (WikiText-2)

```bash
python scripts/run_perplexity.py \
    --db-path weights.duckdb \
    --output results/transql_ppl.json \
    --pivot-width 32 --subquery-width 4 \
    --max-chunks 64 \
    --memory-limit 16GB --threads 4 \
    --temp-directory ./duckdb_tmp
```

### Baseline comparison (without ROW2COL pivot)

```bash
python scripts/run_prefill.py \
    --db-path weights.duckdb \
    --output results/prefill_no_pivot.json \
    --no-pivot \
    --memory-limit 16GB --threads 4 \
    --temp-directory ./duckdb_tmp

python scripts/run_decode.py \
    --db-path weights.duckdb \
    --output results/decode_no_pivot.json \
    --no-pivot \
    --memory-limit 16GB --threads 4 \
    --temp-directory ./duckdb_tmp
```

## Step 4B: TranSQL+ on ClickHouse (paper-portability)

Ports the same TranSQL+ pipeline to ClickHouse to demonstrate the paper's claim that the approach is not DuckDB-specific. The code lives in a parallel subpackage — `transql_plus/clickhouse/` — and the frozen DuckDB path in Step 4 is untouched.

See `reproduction_note.md` D12 for the full dialect gap table (paper's DuckDB-only constructs rewritten for ClickHouse: `PIVOT` → `groupArrayIf`, `POSITIONAL JOIN` collapsed to one `SELECT`, `list_dot_product` → `dotProduct`, etc.) and `results/clickhouse_sql_probe.json` for the raw probe output.

### Bring up ClickHouse (Docker)

```bash
docker run -d -p 9000:9000 -p 8123:8123 --name ch_transql \
    --ulimit nofile=262144:262144 \
    -e CLICKHOUSE_SKIP_USER_SETUP=1 \
    clickhouse/clickhouse-server:latest

pip3 install clickhouse-connect
```

The `CLICKHOUSE_SKIP_USER_SETUP=1` flag leaves the `default` user passwordless so the scripts can connect over HTTP on 8123 without credential setup — acceptable here because the container only binds to localhost.

### (Optional) Re-run the dialect probe

```bash
python scripts/probe_clickhouse_sql.py \
    --ch-host localhost --ch-port 8123 \
    --output results/clickhouse_sql_probe.json
```

Populates `results/clickhouse_sql_probe.json` with one row per construct (status ∈ {`supported`, `workaround-exists`, `unsupported`}). D12's gap table is generated from this output.

### Load weights into ClickHouse

Reuses the chunked CSVs under `weights_csv/` — no re-preprocessing needed. Loads take ~3 min for 32 layers (CSVs streamed via `CSVWithNames`):

```bash
python -m preprocessing.load_weights_clickhouse \
    --csv-dir weights_csv \
    --ch-host localhost --ch-port 8123 --ch-database default \
    --chunk-size 32 --num-layers 32
```

### Benchmarks (same protocol as Step 4)

```bash
# Prefill: 2 warmup + 3 measured runs per prompt length
python scripts/run_clickhouse_prefill.py \
    --ch-host localhost --ch-port 8123 --ch-database default \
    --prompts-dir prompts \
    --output results/clickhouse_prefill.json \
    --lengths 25 50 100 200 \
    --max-memory-usage 16GB --max-threads 4 \
    --pivot-width 32

# Decode: 2 warmup + 49 measured decode steps
python scripts/run_clickhouse_decode.py \
    --ch-host localhost --ch-port 8123 --ch-database default \
    --prompts-dir prompts \
    --output results/clickhouse_decode.json \
    --lengths 25 50 100 200 \
    --max-memory-usage 16GB --max-threads 4 \
    --pivot-width 32

# Perplexity on WikiText-2
python scripts/run_clickhouse_perplexity.py \
    --ch-host localhost --ch-port 8123 --ch-database default \
    --output results/clickhouse_ppl.json \
    --max-chunks 64 \
    --max-memory-usage 16GB --max-threads 4 \
    --pivot-width 32
```

`--max-memory-usage 16GB` (accepts raw bytes or `K`/`M`/`G`/`T` suffix, 1024-based to match DuckDB's `memory_limit`) and `--max-threads 4` match the paper's c7.2xlarge hardware profile (same envelope used for DuckDB in Step 4).

## Step 5: Baselines

### Step 5A: llama.cpp

The paper states "the models used here are unquantized full-precision", so we compare against llama.cpp F32 only (no quantization). **Run llama.cpp under the same 4-core / 16 GB constraint** for an apples-to-apples comparison.

```bash
# Build llama.cpp
cd ~/Project/llama.cpp && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j

# Convert HuggingFace model to GGUF (F32, unquantized)
# (e.g., ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/)
python ~/Project/llama.cpp/convert_hf_to_gguf.py \
    ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/ \
    --outtype f32 \
    --outfile ~/Project/llama.cpp/models/llama-3-8b-f32.gguf
```

### Paper-faithful benchmarks (4 cores, 16 GB RAM)

llama.cpp uses mmap by default, so weights stream from disk — comparable to DuckDB's out-of-core behavior. For strict RAM parity with the paper's 16 GB instance, run under a cgroup.

**llama-bench warmup note.** Unlike TranSQL+'s in-process protocol (2 warmup + 3 measured), `llama-bench` only exposes `--no-warmup` (a boolean) — the warmup *count* is hardcoded by the build (typically 1). We therefore run two distinct configurations and report both:

- **Cold start** = one run after `drop_caches`, `--no-warmup`, `-r 1`. Reveals the storage floor.
- **Warm** = `-r 3` with the build's default (1) warmup included. Reveals the in-RAM compute floor.

Pipe `llama-bench -o json` into `results/llamacpp_f32_{cold,warm}.json` so `scripts/collect_results.py` (Step 6) can parse them:

```bash
mkdir -p results

# --- Cold start (first run, page cache dropped, no warmup) -------------
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 -- \
    ~/Project/llama.cpp/build/bin/llama-bench \
    -m ~/Project/llama.cpp/models/llama-3-8b-f32.gguf \
    -t 4 -p 25 -n 0 -r 1 --no-warmup \
    -o json > results/llamacpp_f32_cold.json

# --- Warm measurements (llama-bench's built-in warmup + -r 3 measured) -
# One invocation produces 4 prefill tests (pp 25/50/100/200) + 1 decode test (tg 50).
# The decode test is prompt-length-independent in llama-bench's model, so the
# collector broadcasts the single tg value to every prompt length.
systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 -- \
    ~/Project/llama.cpp/build/bin/llama-bench \
    -m ~/Project/llama.cpp/models/llama-3-8b-f32.gguf \
    -t 4 -p 25,50,100,200 -n 50 -r 3 \
    -o json > results/llamacpp_f32_warm.json

# Perplexity (requires wikitext-2-raw/wiki.test.raw, see Setup section)
systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 -- \
    ~/Project/llama.cpp/build/bin/llama-perplexity \
    -m ~/Project/llama.cpp/models/llama-3-8b-f32.gguf \
    -t 4 -f wikitext-2-raw/wiki.test.raw \
    2>&1 | tee results/llamacpp_f32_ppl.txt
```

If `systemd-run` is unavailable (containers, macOS), drop the `systemd-run ... --` prefix — `-t 4` still matches the paper's thread count, but without the cgroup RAM is unconstrained; note this when reporting numbers. When quoting results, state your `llama-bench` build's warmup behavior (`llama-bench --help` shows what's available; the build used here only has `--no-warmup`) so the protocol is traceable.

**Storage caveat.** The paper's AWS c7.2xlarge uses network-attached EBS gp3 at ~125 MB/s baseline; a local SATA SSD (~500 MB/s) or NVMe (~3 GB/s) is much faster, so absolute latencies will not match the paper's figures. The *ratio* between TranSQL+ and llama.cpp on the same host is still a fair reproduction of the paper's claim. See `reproduction_note.md` → "llama.cpp Baseline: Settings Parity & Storage Analysis" for the full analysis and the reason we do not add an `IOReadBandwidthMax` throttle.

### Step 5B: DeepSpeed

A second inference-framework baseline (v1 `init_inference` API, CPU BF16; DeepSpeed CPU accelerator does not support float16). Mirrors Step 5A's cold/warm protocol — the script does not drop caches itself; the user drops the page cache before the cold run. Same 4-thread / 16 GiB envelope as all other measurements.

```bash
# Editable install against the local 0.18.9 checkout
pip3 install -e /home/pei/Project/DeepSpeed

# --- Cold run (drop page cache, then launch under 16 GiB cgroup) ------
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 -- \
    python scripts/run_deepspeed.py --cold \
    --threads 4 \
    --lengths 25 50 100 200 \
    --output results/deepspeed_bf16_cold.json

# --- Warm run (same-process warmup, no cache drop) --------------------
systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 -- \
    python scripts/run_deepspeed.py \
    --threads 4 \
    --lengths 25 50 100 200 \
    --output results/deepspeed_bf16_warm.json
```

One invocation produces both prefill (2 warmup + 3 measured) and decode (2 warmup + 49 measured) numbers at every prompt length. If `systemd-run` is unavailable, drop the `systemd-run ... --` prefix — `--threads 4` (propagated to `torch.set_num_threads` + `OMP_NUM_THREADS`) still pins thread count, but RAM is unconstrained; note that when reporting.

## Step 6: Aggregate and Compare

Once Steps 4 and 5 have populated `results/`, produce a single side-by-side table. Run from the project root so the default paths resolve to this directory's `results/`:

```bash
cd /home/pei/Project/reproducility-transql_plus
python scripts/collect_results.py --results-dir results
```

**Inputs** (read from `results/`):
- `prefill.json`, `decode.json`, `transql_ppl.json` — TranSQL+ on DuckDB (Step 4)
- `clickhouse_prefill.json`, `clickhouse_decode.json`, `clickhouse_ppl.json` — TranSQL+ on ClickHouse (Step 4B)
- `llamacpp_f32_cold.json`, `llamacpp_f32_warm.json` — llama-bench `-o json` (Step 5A)
- `llamacpp_f32_ppl.txt` — llama-perplexity text output (Step 5A)
- `deepspeed_bf16_cold.json`, `deepspeed_bf16_warm.json` — DeepSpeed (Step 5B)

**Output** (written to `results/`):
- `combined_results.json` — unified schema, one row per `(system, run_type, prompt_length)`

The script also prints a comparison table to stdout with one column per `(system, run_type)` — e.g. `transql+/warm`, `transql+/ch/warm`, `llamacpp_f32/cold`, `llamacpp_f32/warm`, `deepspeed_bf16/cold`, `deepspeed_bf16/warm` — for prefill/decode latency, throughput, peak RSS, and perplexity.

Notes:
- Rows missing from any input file show `--`; the script is resilient to partial runs.
- llama-bench does not report peak RSS in its JSON, so that column is populated only for the two TranSQL+ backends and DeepSpeed.
- llama.cpp decode is measured from empty context (`-n 50` with `n_prompt=0` in its model); the single tg value is broadcast to every prompt length in the table. This is standard practice when comparing llama-bench numbers — see `reproduction_note.md` → "llama.cpp Baseline".
- To write the combined JSON elsewhere, pass `--output /path/to/file.json`.

## Additional Scripts

### ROW2COL Pivot Tuning (Section 4.3)

```bash
python scripts/tune_pivot.py \
    --chunk-size 32 --threads 4 \
    --lengths 25,50,100,200 \
    --memory-limit 16GB --temp-directory ./duckdb_tmp
```

See Step 2 above for details. Pass the same `--memory-limit` / `--temp-directory` / `--threads` values you use for the full benchmark so the tuned pair transfers correctly. Use `--chunk-size 64` only if exploring a different storage layout.

### Weight-Pivot Cache Microbenchmark (Decision D9)

```bash
python scripts/bench_pivot_cache.py
```

Measures per-run latency of inline PIVOT vs cached PIVOT for one Llama3-8B-sized MatMul and verifies the two produce bitwise-identical results. Used to justify D9's cache + `pivot_setup_time_s` separation.

### Softmax Overflow Demonstration (Decision D5)

```bash
python scripts/demo_softmax_overflow.py
```

Shows that the paper's 2-step softmax (`exp, SUM, div`) overflows for `max(score) > 709`, while the stable 4-step variant handles all cases correctly.

## Project Structure

```
transql_plus/           # Core library (Sections 3-4) — DuckDB path, frozen
  config.py             # Model configuration (D6: no magic numbers)
  sql_templates.py      # SQL code generation (Section 3.2, Table 1)
  compute_dag.py        # Computation DAG (Section 3.2)
  dag_to_sql.py         # DAG to SQL expansion (Section 3.2)
  postopt.py            # Post-optimisations (Section 4)
  runner.py             # Inference runner (Section 5)
  clickhouse/           # ClickHouse dialect port (§D12, parallel subpackage)
    sql_templates_ch.py # ClickHouse templates (dotProduct, arrayMap, ...)
    postopt_ch.py       # CTE merge + fusion + pivot rewrites (§4.1–4.3)
    runner_ch.py        # ClickHouseRunner (same public API as TranSQLRunner)

preprocessing/          # Weight preprocessing (Section 3.1)
  extract_weights.py    # Extract from PyTorch/ONNX + constant folding
  preprocess_weights.py # Algorithm 1: matrix to chunked relational tables
  load_weights_duckdb.py  # Load chunked CSV into DuckDB
  load_weights_clickhouse.py  # Load chunked CSV into ClickHouse (MergeTree)

scripts/                # Measurement and utilities
  run_prefill.py        # Prefill benchmark (DuckDB)
  run_decode.py         # Decode benchmark with KV cache (DuckDB)
  run_perplexity.py     # WikiText-2 perplexity (DuckDB)
  run_clickhouse_{prefill,decode,perplexity}.py  # ClickHouse counterparts
  probe_clickhouse_sql.py  # Dialect probe feeding D12 gap table
  run_deepspeed.py      # DeepSpeed baseline (both prefill and decode)
  sample_prompts.py     # Prompt sampling from LMSys-Chat-1M
  tune_pivot.py         # ROW2COL pivot width tuning (§4.3)
  bench_pivot_cache.py  # D9 weight-pivot cache microbenchmark
  demo_softmax_overflow.py # Decision D5 demonstration

tests/                  # 49 tests
  test_sql_templates.py # Per-operator correctness vs NumPy
  test_postopt.py       # CTE merge, table fusion, pivot
  test_single_layer.py  # Full transformer layer
  test_preprocessing.py # Chunking, CSV, DuckDB round-trip
  test_runner.py        # Runner: prefill, pivot caching, JSON topology
```

## Design Decisions

All design decisions are documented in:
- `reproduction_note.md` — Decisions D1-D12, measurement protocol, known issues (D12 covers the ClickHouse dialect port in Step 4B)
- `comparison_analysis.md` — Gap analysis: reproduction vs AQP_middleware vs paper
