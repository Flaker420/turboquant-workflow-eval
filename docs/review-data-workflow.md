# Review: Data Creation & Methods — Quality and Pertinence

> Scope: evaluation of the data artefacts, measurement methodology, and
> fitness-for-purpose of the test/implementation/benchmarking framework
> in this repository.

---

## 1. Data Creation

### 1.1 Prompt Pack Design

The entire evaluation dataset is a single YAML file
(`prompts/workflow_prompts.yaml`) containing **14 hand-crafted prompts** in four
categories:

| Category   | Count | Example prompt IDs              |
|------------|------:|---------------------------------|
| Reasoning  |     4 | reasoning_01 … reasoning_04    |
| Math       |     4 | math_01 … math_04              |
| Coding     |     4 | coding_01 … coding_04          |
| Retrieval  |     2 | retrieval_01, retrieval_02      |

**Strengths:**

- **Domain relevance.** The prompts mirror tasks that a real user of the
  compressed model would care about: structured reasoning, numerical
  computation, code generation, and factual extraction. They are not generic
  benchmarks pulled from a leaderboard; they are workflow-specific.
- **`watch_for` annotations.** Each prompt carries a short human-readable hint
  (e.g. `"correct numeric reasoning"`, `"working Python and complexity note"`)
  that tells the reviewer what to look for when comparing outputs
  (`prompts/workflow_prompts.yaml:5–6`).
- **Deterministic generation.** The study config sets `temperature: 0.0` and
  `do_sample: false` (`configs/studies/default_qwen35_9b.py:10–11`), which eliminates
  sampling variance and makes policy-to-policy comparisons reproducible.

**Weaknesses:**

- **Small sample size.** Sixteen prompts provide very little statistical power.
  If compression degrades only 10–15% of outputs, a 16-prompt run may not
  surface the failure at all — or may surface it once, giving no basis for
  confidence. The retrieval category has only two prompts, which is
  insufficient to draw any category-level conclusions.
- **All prompts are short.** Every prompt is under ~200 tokens of input. Yet
  the study itself acknowledges that compression artefacts are worst on long
  prompts (`reasoning_03` explicitly mentions "subtle answer drift only on
  long prompts"). The `max_input_tokens` setting of 4096 is never exercised.
  This is a significant gap: the evaluation does not stress-test the regime
  where KV-cache compression matters most.
- **Limited ground-truth coverage.** Math prompts now carry `reference_answer`
  fields and coding prompts carry `test_cases` with input/expected pairs,
  enabling automated pass/fail scoring for those categories. Reasoning and
  retrieval prompts still lack reference outputs and require human review of
  `examples.md`.
- **No multi-turn or system-prompt coverage.** Real workflows frequently
  involve multi-turn conversations and system prompts. Compression artefacts
  can accumulate across turns, but this is untested.
- **Inline retrieval contexts are trivially small.** The two retrieval prompts
  embed ~5-line contexts. A meaningful retrieval stress-test would use
  contexts of hundreds or thousands of tokens to exercise cache capacity under
  compression.

### 1.2 Prompt Loading

`src/turboquant_workflow_eval/prompts.py` supports two sources: a YAML file
(production path) and four hardcoded `"builtin"` prompts used only by the
preflight phase. The loader is straightforward and well-tested
(`tests/test_prompts.py`). No issues here.

### 1.3 Data at Rest

The repository stores **no pre-computed data** — no CSVs, no parquet files, no
cached model outputs. All data is generated at runtime and written to an
`outputs/` directory. This is a clean design choice for an evaluation harness:
it guarantees that results always reflect the current code and model state.

---

## 2. Methodological Rigor

### 2.1 Measurement Approach

**`generate_one()`** (`src/turboquant_workflow_eval/generation.py:35–77`):

| Metric            | How it is measured                        | Sound? |
|-------------------|-------------------------------------------|--------|
| `latency_s`       | `time.perf_counter()` around `model.generate()` + `cuda.synchronize()` | Yes — synchronize ensures GPU work completes before the timer stops. |
| `peak_vram_gb`    | `torch.cuda.max_memory_allocated()` after resetting peak stats | Yes — reset before generation isolates the measurement to this call. |
| `tokens_per_second`| `output_tokens / latency_s`             | Adequate for relative comparison, though it includes prompt-processing time in the denominator. |
| `prompt_tokens`   | Tokenizer output shape                   | Correct. |
| `output_tokens`   | Generated sequence length minus prompt    | Correct. |

**Concerns (all resolved as of PR #8):**

1. ~~**No warmup pass.**~~ **Resolved.** A single untimed warmup generation
   (`generate_one(model, tokenizer, "Say hello.", runtime_cfg)`) now runs
   before the timed prompt loop (`study.py:101`), eliminating JIT/compilation
   noise from the first real measurement.

2. ~~**Single run per prompt.**~~ **Resolved.** The study config now supports a
   `repetitions` parameter (default 3 per `configs/studies/default_qwen35_9b.py:16`).
   Additional repetitions collect latency, tokens/sec, and VRAM measurements
   (`study.py:118–124`), and the results include per-metric mean and standard
   deviation via `_aggregate_stats()` (`study.py:27–36`).

3. ~~**Fixed policy execution order.**~~ **Resolved.** The runtime config
   supports `shuffle_policies` and `shuffle_seed` (`study.py:70–74`). When
   enabled, the policy execution order is randomised with a deterministic seed
   for reproducibility.

### 2.2 Statistical Infrastructure

The `OnlineScalarStats` class (`stats.py:11–52`) implements streaming
mean/variance via sum-of-squares decomposition. The formula
`variance = (sum_sq / n) - mean²` is numerically less stable than Welford's
algorithm for large counts, but for the relatively small tensor counts in a
preflight pass this is acceptable.

`ProjectionTensorStats` (`stats.py:56–88`) captures four meaningful tensor
health metrics: vector L2 norm, absolute values, channel-wise RMS mean, and
channel-wise RMS max. These are well-chosen for detecting compression-induced
distribution shifts in Q/K/V projections.

### 2.3 Preflight Instrumentation

The hook-based tensor capture system (`hooks.py`, `preflight.py`) is cleanly
implemented:

- Forward hooks are registered via a context manager that guarantees cleanup
  (`hooks.py:66–70`).
- Statistics are accumulated online without storing full tensors.
- The preflight report is JSON-serializable and machine-readable.

**No issues found.** This is the strongest methodological component of the
harness.

### 2.4 Adapter & Policy System

The adapter interface (`adapters/base.py`) defines three methods:
`prepare_model`, `describe`, `cleanup`. The baseline adapter
(`adapters/none.py`) is a clean pass-through. The `TurboQuantAdapter`
(`adapters/turboquant.py`) wraps turboquant-core, handling field-name
normalization between the eval harness and the core library.

The safe and aggressive policy templates
(`configs/policies/safe_template.py`, `aggressive_template.py`) are enabled
and point to `TurboQuantAdapter`. The harness can run multi-policy comparison
studies out of the box.

---

## 3. Reporting & Evaluation

### 3.1 Output Formats

The reporting module (`src/turboquant_workflow_eval/reporting.py`) writes five
artefact types:

| File                  | Format    | Purpose                                    |
|-----------------------|-----------|--------------------------------------------|
| `rows.jsonl`          | JSONL     | Machine-readable full records              |
| `workflow_compare.csv`| CSV       | Spreadsheet-friendly metrics table         |
| `examples.md`         | Markdown  | Side-by-side human-readable comparison     |
| `text_outputs/*.md`   | Markdown  | One file per policy×prompt pair            |
| `run_summary.json`    | JSON      | Run metadata (policy count, prompt count)  |

This is a well-designed set of outputs for a manual review workflow.

### 3.2 Automated Evaluation (rewritten in the divergence-metrics PR)

The evaluation layer originally landed as a green/yellow/red verdict
aggregator over latency, semantic-similarity, math, and code-runner
signals. That aggregator was deleted in the divergence-metrics rewrite
(commit `90fb869` and follow-ups) because it was measuring the wrong
thing for this project: a turboquant policy whose generated tokens were
byte-identical to baseline could still be flagged "yellow" on latency
jitter alone. The current scoring layer captures two direct metric
families instead -- both computed in the post-hoc `score_results` join,
both unit-tested torch-free:

- **Token-level divergence vs baseline.** `scoring.py:compute_divergence`
  (lines 175–206) compares each non-baseline row's `output_token_ids`
  against the baseline row for the same prompt and returns
  `exact_match`, `first_divergence_token`, `common_prefix_tokens`,
  `common_prefix_frac`, `token_edit_distance` (plain O(n*m) Levenshtein
  on token IDs), and `output_length_delta_tokens`. Sentinel
  `first_divergence_token == -1` is reserved for the baseline row
  itself; sequences that are strict prefixes of one another return
  `first_divergence_token == min(len(policy), len(baseline))`.

- **Theoretical KV-cache compression bytes.**
  `scoring.py:compute_kv_cache_bytes` (lines 237–320) is a pure-Python
  helper that takes the per-row policy knobs (`bit_width`,
  `residual_window`, `compressible_layers`, `compressible_heads`,
  `key_strategy`, `value_strategy`) plus model topology
  (`num_hidden_layers`, `num_key_value_heads`, `head_dim`) and the
  per-row `prompt_tokens + output_tokens` count, and returns
  `kv_cache_bytes_baseline`, `kv_cache_bytes_policy`,
  `kv_cache_compression_ratio`, and `kv_cache_bytes_saved`. Per-element
  cost is modelled as `bit_width / 8` so future bit-packing work
  plugs into the same accounting helper without changing callers.
  `key_strategy == "mse+qjl"` adds a crude one-byte-per-compressed-K-token
  side-channel overhead.

- **Pareto summary.** `reporting.py:summarize_divergence` (lines
  275–323) folds the per-row metrics into a per-policy
  `divergence_summary` block in `run_summary.json` -- exact-match
  count/rate, mean common-prefix fraction, mean / max edit distance,
  mean compression ratio, mean bytes saved. The next PR can plot
  `(exact_match_rate, mean_kv_cache_compression_ratio)` straight off
  this block.

The legacy `extract_numbers`, `check_reference_answer`, and
`compute_semantic_similarity` helpers still live in `scoring.py` for
diagnostic side-channels but no longer drive any row-level decision;
the math and code-runner helpers are similarly available but not wired
into the post-hoc scoring path.

**Backbone for these metrics**:
[`vendor/turboquant-core/docs/algorithm-comparison.md`](../vendor/turboquant-core/docs/algorithm-comparison.md)
§"PR1: Algorithmic Ablation Matrix" / "Measurement discipline" -- the
audit-of-eight community implementations enumerates exactly the per-row
bookkeeping any honest ablation must carry (compressed-token count,
residual-window length, protected-layer policy, honest bytes/token
including all metadata, end-to-end memory reduction). Every row this
harness writes carries that bookkeeping by construction.

**Remaining gaps:**

- Reasoning and retrieval prompts still lack reference outputs and
  require human review of `examples.md` and `watch_for` annotations
  alongside the divergence numbers.
- The harness has no long-context prompts in the default pack, so
  realistic compression ratios (where the residual window can no
  longer cover the whole sequence) are not yet exercised. This is the
  next PR.

---

## 4. Test Coverage

The test suite (`tests/`) validates scaffolding — config loading, prompt
parsing, slugification, and output-file creation. These are valuable for
maintaining the harness itself.

**Not tested:**

- The study orchestration pipeline end-to-end (would require a model or mock).
- The generation function with a real or mock model.
- The preflight hook registration and statistics accumulation on a real model.
- Any form of output quality assertion.

The tests ensure the **plumbing** works, but do not test the **measurement
logic** or the **evaluation methodology**.

---

## 5. Pertinence Assessment

**The stated goal** (from `docs/study-scope.md`):
> *Which compression policy is usable in my workflow, and what degrades when I
> push harder?*

**Can the current harness answer this?**

| Aspect                  | Answer | Explanation |
|-------------------------|--------|-------------|
| "Which policy is usable" | Yes | The harness reports `exact_match_rate` and per-row `token_edit_distance` against the baseline policy, plus performance metrics (latency, VRAM, throughput). A policy whose `exact_match_rate` is 1.0 is byte-identical to baseline by definition; lower rates with small mean edit distances indicate localised divergence the human reviewer can read in `examples.md`. Reasoning and retrieval prompts still benefit from human review of `watch_for` annotations alongside the divergence numbers. |
| "What degrades"         | Yes (for token output) | `first_divergence_token` and `token_edit_distance` describe exactly where and how much each policy's output drifts from baseline. Memory degradation is captured analytically by `kv_cache_compression_ratio` -- 1.0 means the policy fell back to baseline (e.g. residual window covered the whole sequence), values > 1.0 quantify the savings. Side-by-side markdown comparison remains available for human review. |
| "When I push harder"    | Weak | The prompt set does not vary in difficulty or length. There are no long-context, multi-turn, or adversarial prompts that would stress-test compression at its limits, so `kv_cache_compression_ratio` typically reads ~1.0 on the default pack -- the residual window covers everything. `scripts/generate_prompts.py` exists to create long-context prompts but they are not in the default prompt pack. |

**Overall pertinence: good.** The harness now has a substantive analysis layer
on top of its data-collection framework. The primary remaining gap is **data
coverage** — the prompt set is too small and too short to stress-test the regime
where KV-cache compression matters most.

---

## 6. Recommendations

Ranked by impact-to-effort ratio. Implementation status updated as of PRs
#8–#10.

### High Impact, Low Effort

1. **~~Add a warmup generation.~~** **Done.** `study.py:101` runs an untimed
   warmup generation before the timed prompt loop.

2. **~~Add reference answers for math prompts.~~** **Done.**
   `scoring.py:check_reference_answer()` compares extracted numbers against
   `reference_answer` fields in the prompt YAML (with 5% relative tolerance).

3. **~~Add output-length delta to CSV.~~** **Done.** `study.py:199–202`
   computes `output_length_delta_pct` for each row relative to the baseline.

### High Impact, Moderate Effort

4. **Add 8–12 long-context prompts.** **Partial.**
   `scripts/generate_prompts.py` can create long-context prompts
   programmatically, and `configs/studies/full.py` references a generated
   prompt pack alongside the fixed pack. However, no long-context prompts are
   included in the default prompt pack yet.

5. **~~Execute generated code for coding prompts.~~** **Reverted along
   with the verdict pipeline.** `code_runner.run_code_with_tests` is
   still importable from `src/turboquant_workflow_eval/code_runner.py`
   but is no longer called from the study loop or the scoring layer:
   the divergence-metrics rewrite removed the `code_passed`,
   `code_failed`, `code_errors`, and `code_verdict` fields from the
   row schema along with the green/yellow/red verdict aggregator that
   consumed them. Re-enabling code execution would require a new
   scoring-layer contract consistent with the divergence-metrics
   schema.

6. **~~Add 3–5 repetitions per prompt-policy pair.~~** **Done.** The study
   config supports a `repetitions` parameter (default 3). Additional
   repetitions collect latency/TPS/VRAM metrics (`study.py:118–124`) and
   report mean ± std. Only one text output is kept per pair.

### Medium Impact, Higher Effort

7. **~~Implement automated semantic comparison.~~** **Reverted in the
   divergence-metrics PR.** `scoring.py:compute_semantic_similarity()` is
   still in the module for diagnostic side-channels but no longer feeds
   any row-level decision. Community evidence
   ([algorithm-comparison.md §"Measurement discipline"](../vendor/turboquant-core/docs/algorithm-comparison.md))
   indicated that semantic-similarity scoring was the wrong axis when the
   project's actual goal is byte-identical output -- a fuzzy embedding
   distance can flag exact matches as "drifted" or vice versa. Replaced
   by direct token-level divergence
   (`scoring.py:compute_divergence`, lines 175–206).

8. **~~Implement the green/yellow/red threshold system.~~** **Reverted in
   the divergence-metrics PR.** `scoring.py:compute_verdict()` and the
   accompanying `ThresholdsConfig` dataclass are deleted entirely.
   Community evidence indicated the verdict aggregator was measuring the
   wrong thing for byte-identical output runs -- it routinely flagged
   exact-match policies as "yellow" on latency jitter alone. Replaced by
   direct token-level divergence + theoretical KV-cache-bytes metrics
   computed in the post-hoc `score_results` join. See
   [algorithm-comparison.md §1 "Quantization Core: MSE vs. MSE+QJL" and
   "Measurement discipline"](../vendor/turboquant-core/docs/algorithm-comparison.md)
   for the conceptual backbone.

9. **~~Randomize or interleave policy execution order.~~** **Done.**
   `study.py:70–74` implements `shuffle_policies` with a deterministic
   `shuffle_seed` in the runtime config.

10. **Add multi-turn prompt sequences.** **Not started.** The `PromptSpec`
    dataclass includes a `turns` field and `generate_one()` supports
    multi-turn generation, but no multi-turn prompts exist in the prompt pack
    yet.

---

## Summary

| Dimension              | Rating       | Notes |
|------------------------|--------------|-------|
| Prompt relevance       | Good         | Domain-appropriate, workflow-specific prompts |
| Prompt coverage        | Insufficient | Too few (14), too short, no long-context stress tests |
| Measurement quality    | Good         | Correct timing, VRAM tracking, deterministic config |
| Statistical rigour     | Adequate     | Warmup pass, 3 repetitions with mean/std, optional policy shuffle |
| Automated evaluation   | Good         | Token-level divergence vs baseline (`exact_match`, `first_divergence_token`, `token_edit_distance`) and theoretical KV-cache compression bytes derived from policy settings -- both computed in the post-hoc `score_results` join |
| Reporting              | Good         | Multiple formats, clean structure, divergence + KV-cache columns in all outputs, per-policy `divergence_summary` block in `run_summary.json` |
| Fitness for purpose    | Good         | Strong collection and analysis framework; remaining gap is data coverage |

The harness has matured from a data-collection scaffold into a substantive
evaluation framework. Recommendations 1–6 and 9 are still implemented;
recommendations 7–8 (semantic similarity scoring and the
green/yellow/red verdict aggregator) were intentionally **reverted** in
the divergence-metrics PR after community evidence indicated they were
measuring the wrong axis for byte-identical output runs -- replaced by
the direct token-level divergence + KV-cache-bytes metrics described
above. The primary remaining gap is **data coverage**: the prompt set
is too small and too short to stress-test the regime where KV-cache
compression matters most (recommendations 4 and 10), so
`kv_cache_compression_ratio` typically reads ~1.0 on the default pack.
