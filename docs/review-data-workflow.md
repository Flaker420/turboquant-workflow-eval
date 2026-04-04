# Review: Data Creation & Methods — Quality and Pertinence

> Scope: evaluation of the data artefacts, measurement methodology, and
> fitness-for-purpose of the workflow study implemented in this repository.

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
  `do_sample: false` (`configs/studies/default.yaml:10–11`), which eliminates
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
   `repetitions` parameter (default 3 per `configs/studies/default.yaml:16`).
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
(`adapters/none.py`) is a clean pass-through. The external stub
(`adapters/external_stub.py`) fails loudly if enabled without a real backend.

**Observation:** The safe and aggressive policy templates
(`configs/policies/safe_template.yaml`, `aggressive_template.yaml`) are disabled
stubs pointing to `ExternalCompressionAdapterStub`. This means the harness
**currently can only run a single-policy (baseline) study**. Until real
compression backends are wired in, the comparison framework is untestable for
its core purpose.

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

### 3.2 Automated Evaluation (implemented as of PRs #8–#10)

The evaluation layer originally identified as absent has been implemented:

- **Math reference-answer checking.** `scoring.py:check_reference_answer()`
  extracts numbers from model output and compares them against the
  `reference_answer` field on each prompt with configurable tolerance (default
  5% relative). Results appear as `math_correct` in the output rows.

- **Code execution for coding prompts.** `code_runner.py` extracts Python code
  blocks from model output and runs them against `test_cases` defined in the
  prompt YAML. Results include `code_passed`, `code_failed`, `code_errors`,
  and a `code_verdict` (pass / fail / error).

- **Semantic similarity.** `scoring.py:compute_semantic_similarity()` uses
  sentence-transformers (optional dependency) to compute cosine similarity
  between baseline and compressed outputs. The score appears as
  `semantic_similarity` in the output rows.

- **Output-length delta.** `study.py:199–202` computes
  `output_length_delta_pct` for each row relative to the baseline policy.

- **Configurable green/yellow/red verdict system.**
  `scoring.py:compute_verdict()` aggregates latency regression, output-length
  delta, semantic similarity, math correctness, and code execution results
  into a single `verdict` per row. Thresholds are configurable in the study
  config (`configs/studies/default.yaml:19–25`).

**Remaining gaps:**

- Reasoning and retrieval prompts still lack reference outputs and require
  human review of `examples.md` and `watch_for` annotations.
- No automated text diff between baseline and compressed outputs beyond the
  semantic similarity score.

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

| Aspect                  | Verdict | Explanation |
|-------------------------|---------|-------------|
| "Which policy is usable" | Yes | The harness collects performance metrics (latency, VRAM, throughput) and automated quality signals (math correctness, code execution, semantic similarity). The verdict system aggregates these into a green/yellow/red decision per prompt-policy pair. Reasoning and retrieval prompts still require human judgment. |
| "What degrades"         | Mostly | Automated degradation detection via output-length delta, semantic similarity, and the verdict system. Side-by-side markdown comparison remains available for human review. |
| "When I push harder"    | Weak | The prompt set does not vary in difficulty or length. There are no long-context, multi-turn, or adversarial prompts that would stress-test compression at its limits. `scripts/generate_prompts.py` exists to create long-context prompts but they are not in the default prompt pack. |

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
   programmatically, and `configs/studies/full.yaml` references a generated
   prompt pack alongside the fixed pack. However, no long-context prompts are
   included in the default prompt pack yet.

5. **~~Execute generated code for coding prompts.~~** **Done.**
   `code_runner.py` extracts Python code blocks and runs them against
   `test_cases` from the prompt YAML in a sandboxed subprocess with timeout.
   Results appear as `code_passed`, `code_failed`, `code_errors`, and
   `code_verdict` in output rows.

6. **~~Add 3–5 repetitions per prompt-policy pair.~~** **Done.** The study
   config supports a `repetitions` parameter (default 3). Additional
   repetitions collect latency/TPS/VRAM metrics (`study.py:118–124`) and
   report mean ± std. Only one text output is kept per pair.

### Medium Impact, Higher Effort

7. **~~Implement automated semantic comparison.~~** **Done.**
   `scoring.py:compute_semantic_similarity()` uses sentence-transformers
   cosine similarity between baseline and compressed outputs. Requires the
   optional `sentence-transformers` dependency.

8. **~~Implement the green/yellow/red threshold system.~~** **Done.**
   `scoring.py:compute_verdict()` uses configurable thresholds from the study
   config (`latency_yellow_pct`, `latency_red_pct`, `similarity_yellow`,
   `similarity_red`, `output_length_yellow_pct`, `output_length_red_pct`).
   The `verdict` column appears in all output formats.

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
| Automated evaluation   | Good         | Math checking, code execution, semantic similarity, configurable verdict system |
| Reporting              | Good         | Multiple formats, clean structure, verdict column in all outputs |
| Fitness for purpose    | Good         | Strong collection and analysis framework; remaining gap is data coverage |

The harness has matured from a data-collection scaffold into a substantive
evaluation framework. Recommendations 1–3 and 5–9 have been implemented,
adding automated quality scoring, statistical repetitions, and a configurable
verdict system. The primary remaining gap is **data coverage**: the prompt set
is too small and too short to stress-test the regime where KV-cache compression
matters most (recommendations 4 and 10).
