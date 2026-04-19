# Reproduction Notes — TranSQL+ (arxiv 2502.02818)

Tracks all design decisions, paper-silent choices, and known gaps.

---

## Design Decisions

### D1. 1D Norm Schema — No row_index
**Paper**: Algorithm 1 only shows 2D chunking. Silent on 1D norm weights.
**AQP_middleware**: `(row_id=0, chunk_id, v)` — redundant constant column.
**Reproduction**: `(chunk_index, v)` — no `row_index`. More efficient schema.
**Rationale**: The paper doesn't prescribe 1D storage. Removing the constant `row_id=0` saves space and simplifies JOINs in RMSNorm.

### D2. RoPE as Single SQL Step
**Paper**: Not mentioned explicitly. The 5 categories (elem-wise arith + shape manip) can cover it but would require 4+ intermediate tables per layer.
**Reproduction**: Single SQL step using `list_transform` with even/odd split and cos/sin rotation.
**Rationale**: Decomposing into separate shape-manip + elem-wise steps would add ~128 intermediate tables across 32 layers for no benefit. The single step is correct and efficient.

### D3. DuckDB Function Mapping
**Paper**: Abstract notation — `DOT(c1, c2)`, `exp(x)`, etc.
**Reproduction**: Maps to DuckDB concrete functions:
- `DOT(c1, c2)` -> `list_dot_product(c1, c2)`
- Element-wise ops -> `list_transform(generate_series(1, len(v)), i -> ...)`
- Array packing -> `array_agg(val ORDER BY col)`

### D4. Causal Mask
**Paper**: Not mentioned explicitly. Required for correct autoregressive attention.
**Reproduction**: `k.row_index <= q.row_index` JOIN condition in `qk_attn_sql()`.
**Rationale**: Standard causal masking. Without it, future tokens leak into attention.

### D5. 2-Step Softmax (Paper-Faithful)
**Paper Table 1**: `Normalize_{exp, SUM, div}` — 2 steps: (1) exp + sum, (2) divide.
**AQP_middleware**: 4-step numerically stable: max, exp(x-max), sum, divide.
**Reproduction**: Default is paper's 2-step. `stable=True` option for 4-step.

**Numerical risk**: The 2-step softmax computes `exp(score)` directly. For sequences longer than ~20 tokens, attention scores can exceed ~88 (the float32 exp overflow threshold), producing `Inf` and then `NaN` after division. The stable variant subtracts `max(score)` first, keeping all exponents <= 0.

**Decision**: Keep 2-step as default to match the paper exactly. Use `stable=True` for real-model inference. See `scripts/demo_softmax_overflow.py` for empirical demonstration.

### D6. Dimensions from ModelConfig
**Paper**: Dimensions derived from ONNX graph or model config.
**AQP_middleware**: Hardcoded `14336`, `4096` in several places (e.g., `transql_postopt.cpp:457,491`).
**Reproduction**: All dimensions flow through `ModelConfig` — no magic numbers.

### D7. Raw Offset chunk_index
**Paper Algorithm 1, line 10**: `chunk_index: c` where `c` steps by `chunk_size` (0, 32, 64, ...).
**AQP_middleware**: Sequential chunk_id (0, 1, 2, ...).
**Reproduction**: Raw offset per Algorithm 1.
**Note**: Both conventions produce correct results — the JOIN key semantics are identical. The reproduction follows the paper's literal pseudocode. Re-chunk uses `out_col - (out_col % chunk_size)` which naturally produces raw offsets.

### D8. Flag Column for Table Fusion
**Paper Section 4.2**: "The weight tables for W_Q, W_K, and W_V are vertically merged using UNION ALL, with a flag column distinguishing the projection type. During SQL generation, this flag becomes part of the shared dimensions, allowing all three vectors to be computed in a single SQL query."
**AQP_middleware**: Uses integer range filter on `out_col` (e.g., `WHERE out_col < 4096` for Q, `BETWEEN 4096 AND 5119` for K). No flag column.
**Reproduction**: Explicit flag column (`'Q'`/`'K'`/`'V'` for QKV; `'G'`/`'U'` for gate+up) per paper.

**Performance note**: The integer-range approach avoids string comparison overhead in the GROUP BY and WHERE clauses. For Llama3-8B dimensions, this could be ~5-10% faster for the fused MatMul step. However, the flag column is what the paper describes, so we follow it for faithfulness.

### D9. Weight Pivot Caching
**Paper**: Section 4.3 describes ROW2COL pivoting but does not explicitly discuss caching pivoted weight tables across inference runs. The example SQL shows `A_pivot CROSS JOIN B_pivot` as referenced tables — silent on whether they are inline CTEs or pre-materialized.
**Reproduction**: Pivot each weight table once during `init()` as a TEMP TABLE and reuse across all `run_prefill()` / `run_decode_step()` calls. `pivot_setup_time_s` is measured and reported **separately** from per-run latency so readers can see both the one-time cost and the amortized per-run cost.
**Rationale**: Weight tables are static. Empirical measurement (`scripts/bench_pivot_cache.py`, one MatMul at Llama3-8B dims — 4096 × 4096, chunk_size=32, seq_len=25, 4 threads, 16GB memory_limit):

| Variant | Per-run latency | Setup |
|---|---|---|
| A — inline PIVOT in every query | 0.77s (+/- 0.07) | 0 |
| B — cached PIVOT (TEMP TABLE once) | 0.004s (+/- 0.000) | 0.94s once |

Caching gives a 192× per-call speedup; break-even at 1.22 runs. Correctness verified: bitwise-identical dot-product results between variants (`Max |A - B| = 0.0`).

**Implementation note (April 2026)**: Earlier revision had a latent bug — `runner._pivot_weight_tables()` created `{weight}_piv` TEMP TABLEs, but the §4.1 CTE-merge pass re-emitted `{weight}_piv AS (PIVOT ...)` as a CTE inside every merged `WITH` block, which shadowed the TEMP TABLE and re-computed the pivot on every call. Fixed by:
1. Adding `iter_pivot_specs(dag, opts)` in `postopt.py` that enumerates exactly which weight pivots will be needed (respecting table-fusion decisions — Q/K/V and gate/up are skipped because fusion uses UNION ALL, not pivoting).
2. Running `_pivot_weight_tables()` **before** `postopt_dag_to_sql()`, passing the set of already-materialized pivot names as `cached_weight_pivots=` so the generator suppresses the redundant pivot step inside merged CTE blocks.
3. Timing step 1 as `runner.pivot_setup_time_s` and surfacing it in all three measurement scripts' output JSON.

### D11. QKV Fusion × Decode KV Cache Interaction

**Paper Section 4.2 (fusion)**: W_Q, W_K, W_V are vertically merged via UNION ALL and the three projections are computed in a single SQL query distinguished by a flag column. The fused output is a single table.

**Paper Section 5 (decode)**: Autoregressive decoding maintains a KV cache — per-layer `l{l}_k_rope` and `l{l}_v` tables — that grows by one row per decode step via `INSERT INTO`.

**Gap**: Neither the paper nor AQP_middleware describes how these two interact. With fusion on, the per-layer V output is absorbed into `l{l}_q_qkv`; downstream consumers (`AttnVMul`) receive it as a filter CTE (`WHERE flag = 'V'`) prepended to their `WITH` block. No standalone `l{l}_v` table exists post-prefill. But `run_decode_step()` does `INSERT INTO l{l}_v ...`, which raises `CatalogException`. The DAG builder marks the V node with `shared=True,  # used by AttnVMul` (`compute_dag.py:149`) — an explicit materialization intent that fusion silently overrides.

**Reproduction**: After prefill (and before the first decode step), extract `l{l}_v` from the fused `l{l}_q_qkv` table via `CREATE TEMP TABLE l{l}_v AS SELECT row_index, chunk_index, v FROM l{l}_q_qkv WHERE flag = 'V'`. Idempotent — runs once, guarded by `self._kv_cache_prepared`. Reset alongside `_drop_inference_tables()` so subsequent prefill→decode cycles re-materialize.

**Cost**: One CTAS per layer of `seq_len × kv_chunks` rows (e.g., 25 × 8 = 200 rows at prompt len 25). Microseconds per layer, <<1% of prefill time. Net vs. disabling fusion: fusion saves 3 materializations/layer (Q, K, V); this restores only V, so still a net win.

**Alternative rejected**: Modifying fusion's `_try_fusion` to force-materialize the V output as a side effect would be more principled but require threading a "preserve these outputs" signal into the CTE-merge phase. Keeping the fix in `runner` localizes the blast radius.

**AQP_middleware alignment**: `comparison_analysis.md:10` notes "after prefill, `l{l}_k_rope` and `l{l}_v` tables persist as KV cache" — confirming the reference behavior that the reproduction now also exhibits.

### D10. DeepSeek MoE — TopKRouting Gap
**Paper Section 3.2**: Claims the 5 core operator categories cover MOE architectures. Mentions "builds indexes on expert identifiers."
**Analysis**: ExpertFFN (matmul + swiglu + rechunk) and MoeAggregate (weighted sum + rechunk) fully decompose into the 5 categories. However, **TopKRouting requires `ROW_NUMBER() OVER (PARTITION BY act_row ORDER BY val DESC)` + `WHERE rk <= K`** — a ranked window function for top-k selection. This is NOT expressible as any of: (1) matrix multiplication, (2) element-wise functions, (3) element-wise arithmetic, (4) shape manipulation, or (5) normalisation.
**Decision**: Not implementing MOE operators. The paper's "5 categories cover MOE" claim appears to be an approximation — the routing/gating step requires a genuinely new primitive (top-k ranked selection via window function).

---

## Measurement Protocol

### Hardware Constraint (Paper Section 5.1)

**Paper**: "We conduct experiments on an AWS c7.2xlarge instance with **4 physical CPU cores and 16GB RAM**—comparable to typical personal devices. Our focus is on evaluating TranSQL+ under memory-constrained conditions."

**Key implication**: The paper's DB (21.3 GB) is **bigger than its RAM (16 GB)**. Spilling to disk / streaming from storage is the intended operating mode, not a bug. The paper explicitly states: "We use DuckDB as our backend, a lightweight analytical database supporting **out-of-core execution**."

**For faithful reproduction on a larger host**, measurement scripts must constrain DuckDB to the paper's resources:

```
python scripts/run_prefill.py --db-path weights.duckdb \
    --memory-limit 16GB --threads 4 \
    --temp-directory ./duckdb_tmp \
    --prompts-dir prompts
```

- `--memory-limit 16GB`: matches the paper's 16 GB RAM instance
- `--threads 4`: matches c7.2xlarge's 4 physical cores
- `--temp-directory`: explicit path for out-of-core spill files

**llama.cpp baseline must match**: run `llama-bench` with `--threads 4` (aka `-t 4`) on the same machine. llama.cpp uses mmap by default so weights stream from disk — comparable to DuckDB's out-of-core behavior. For strict RAM parity, run under a cgroup:

```
systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 -- \
    llama-bench -m llama-3-8b-f32.gguf -t 4 -p 25,50,100,200 -n 0 -r 3
```

**On our 62 GB / 12-core server**: without the constraint flags, measurements would reflect a much more generous memory/compute regime than the paper's 4-core/16 GB setup. Always pass them when reproducing published numbers.

### Warmup Analysis
**AQP_middleware bug**: `run_prefill.py` does `--repeat 3` with NO warmup — run 1 hits cold DuckDB page cache (weight pages not yet in buffer pool), inflating the reported mean.
**llama-bench**: Internally does 5 warmup iterations (hardcoded in llama.cpp source) before measuring `-r 3` runs.
**DuckDB behavior**: After 1 full prefill run, all weight table pages are loaded into the buffer pool and remain there as long as the connection stays open. Unlike CPU cache effects that vary per-run, DuckDB's buffer pool is stable once populated.

### Protocol
```
Prefill (time to first token):
  1. Open DuckDB connection
  2. Pre-pivot all weight tables (timed separately as one-time setup cost)
  3. 2 warmup runs (discard) — warms DuckDB buffer pool
  4. 3 measured runs
  5. Report: mean, std of 3 measured runs

Decode (time per output token):
  1. Run prefill (warms page cache for weight tables)
  2. 2 warmup decode steps (discard) — warms decode query patterns
  3. N measured decode steps
  4. Report: mean, std of measured steps (excluding warmup)
```

**Why 2 warmup**: 1 is sufficient for DuckDB page cache, but 2 provides safety margin and DuckDB may also have JIT/compilation overhead on first execution of certain query patterns. This matches the spirit of llama-bench's warmup without the 5-iteration overkill.

---

## llama.cpp Baseline: Settings Parity & Storage Analysis

### Settings parity

Both stacks run under the paper's c7.2xlarge profile. Equivalences enforced:

| Setting | TranSQL+ | llama.cpp |
|---|---|---|
| Threads | `--threads 4` | `-t 4` |
| Memory cap | DuckDB `--memory-limit 16GB` | cgroup `MemoryMax=16G` |
| No swap | (not applicable; DuckDB spills to `--temp-directory`) | `MemorySwapMax=0` |
| Warmup | 2 discarded runs (reproduction protocol) | built-in 1 warmup (count not tunable; only `--no-warmup` boolean exposed) |
| Measured runs | 3 | `-r 3` |
| Cold-start | drop page cache + 1 run (reported separately) | drop page cache + `-r 1 --no-warmup` |
| Precision | FLOAT32 in `weights.duckdb` | F32 GGUF (no quantization) |
| Storage | `weights.duckdb` on host SSD | `.gguf` (mmap) on same host SSD |

Both stacks thus see the same 4-core / 16 GB / out-of-core regime from the same underlying filesystem. Storage is the one dimension we cannot force into parity with the paper (see below).

### Observed numbers vs paper

Local measurement (2026-04, host: 62 GB RAM / 12 cores / SATA SSD on `/dev/sda`; both stacks cgroup-capped to 4 threads / 16 GB / no swap):

| Workload | Local llama.cpp F32 | Local TranSQL+ (our run) | Paper llama.cpp F32 |
|---|---|---|---|
| Prefill, prompt=25 | ~1.5 s (pp=16.22 t/s) | ~11.6 s | > 100 s (Fig. 6) |
| Decode, per token | ~0.92 s (tg=1.09 t/s) | ~1.36 s | > 15 s (Fig. 7) |

llama.cpp is ~65× faster on our host than on the paper's reported AWS run; TranSQL+ shows the same order-of-magnitude gap. The relative ordering (TranSQL+ slower than llama.cpp at prefill, comparable at decode) matches the paper; the absolute scale does not.

### Storage hypothesis

Llama3-8B in FLOAT32 is ~30 GB of weights. With a 16 GB buffer pool / RAM cap, the remainder must be read from disk on every full pass. Bandwidth dominates wall time once RAM is saturated:

| Storage | Sustained read | Time to read 30 GB | Consistent with |
|---|---|---|---|
| AWS EBS gp3 (paper) | ~125 MB/s baseline, bursty | ~240 s | Paper's >100 s prefill |
| SATA SSD (our host, `/dev/sda`) | ~500 MB/s | ~60 s | Our ~1-11 s (weights partly cached after warmup) |
| NVMe (common research hosts) | ~3,000 MB/s | ~10 s | Much faster than paper |

The paper's AWS c7.2xlarge uses a network-attached EBS volume with gp3's 125 MB/s baseline and a token-bucket burst model — comparable in magnitude to an HDD for sustained reads. After the buffer pool fills, every subsequent miss pays the ~125 MB/s tax. That matches the paper's >100 s prefill far better than any compute-bound explanation.

On our SATA SSD the sustained read ceiling is roughly 4× higher and, crucially, the page cache stays warm across warmup runs (llama.cpp uses `mmap`, so re-reads hit OS cache; DuckDB's buffer pool likewise stays populated across repeats inside one process). After warmup, only residual I/O is needed.

### Decision — no artificial I/O throttle

We considered wrapping both stacks in `systemd-run --scope -p "IOReadBandwidthMax=/dev/sda 125M"` to emulate EBS gp3's baseline. Rejected for this reproduction:

- A cgroup v2 bandwidth cap on a local SSD is **not** equivalent to EBS gp3. EBS has burst credits (3,000 IOPS / 125 MB/s baseline with a credit bucket), per-request round-trip latency over the network, and throughput that varies with queue depth. A flat `IOReadBandwidthMax` matches neither the steady-state nor the burst behavior.
- Adding a throttle reproduces a *simulated bottleneck*, not the paper's actual setup. The faithful setting knobs (threads, memory cap, swap off, out-of-core) are what the paper actually specifies; storage is an AWS artifact.
- Both stacks are measured on the same local storage, so **ratios are fair** even when absolute numbers are not — which is what the paper's figures compare (TranSQL+ vs llama.cpp under identical conditions).
- Readers who want to probe the paper's claim directly can apply the same throttle to both stacks on their own hardware; the same `systemd-run --scope -p "IOReadBandwidthMax=/dev/<your-block-device> 125M"` block works for `llama-bench` and for `scripts/run_prefill.py` / `run_decode.py` / `run_perplexity.py`, and both must be throttled together for the comparison to stay apples-to-apples.

### What to report

When citing our measurements:
- State the host storage (e.g. "SATA SSD, /dev/sda") alongside CPU / RAM, because absolute latencies are storage-bound above the 16 GB cap.
- Compare TranSQL+ vs llama.cpp *as a ratio* on the same host — this is the paper's actual claim and is unaffected by the storage gap.
- Report the cold-start (page-cache-dropped) timing separately from the warm mean. The cold-start reveals the storage floor; the warm mean reveals in-RAM compute.

---

## Known Issues

### Database Size Discrepancy
**Paper Section 3**: "Llama3 8B grows from 16.1GB (raw) to 21.3GB in the database."
**Actual**: raw safetensors (bfloat16) = 16.1GB ✓; weights.duckdb (FLOAT[32] = float32) = ~35GB.

**Investigation (April 2026)**:
- DuckDB 1.5.2 already applies **ALPRD** (its best lossless float32 compressor) to the `v FLOAT[32]` column by default — confirmed via `PRAGMA storage_info`. 35 GB is the lossless floor.
- `FORCE CHECKPOINT` on an already-loaded DB yields **0% reduction** (confirmed in `scripts/test_checkpoint_compression.py`). Compression is applied during the initial write, not at checkpoint time.
- `Chimp` / `Chimp128` compression is **deprecated** in DuckDB 1.5.2; `force_compression='chimp'` rejected by the parser.
- DuckDB has no native `FLOAT16` / `HALF` type. PR #16395 (merged Feb 2025) adds Parquet FLOAT16 *read* support only — values are converted to `FLOAT` on load. There is no `FLOAT16[]` storage.
- The 21.3 GB claim (1.32× over 16.1 GB bfloat16 raw) is only consistent with 16-bit-wide storage. The paper never explains how 21.3 GB is achieved for "unquantized full-precision" weights.

**Conclusion**: The 21.3 GB number remains unexplained. Possibilities: (a) the paper silently bit-packed bfloat16 into `USMALLINT[32]` with query-time reconstruction, (b) a measurement artifact (e.g., snapshot before full materialization), or (c) a different DuckDB version with different storage. AQP_middleware's own database is also ~34 GB with identical `FLOAT[32]` storage, so our 35 GB matches the only other public reproduction exactly.

**Practical implication for reproduction**: do not chase the 21.3 GB target — instead, match the paper's **measurement conditions** (4 cores, 16 GB RAM via `--memory-limit 16GB --threads 4`). Under these constraints, both our 35 GB DB and the paper's 21.3 GB DB are out-of-core (exceed RAM), and DuckDB's out-of-core execution handles both similarly.

### Perplexity Mismatch (from AQP_middleware)
AQP_middleware measured TranSQL+ PPL = 1521.99 vs llama.cpp F32 PPL = 6.51 (234x worse). This indicates a correctness bug in the SQL inference implementation, likely from float32 accumulation error over 32 layers. The diagnostic scripts (`diag_layers.py`, `test_precision_fix.py`) were investigating this. To be monitored in this reproduction.
