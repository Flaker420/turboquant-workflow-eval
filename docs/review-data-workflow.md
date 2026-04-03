# Review: Data Creation & Methods — Quality and Pertinence

> Scope: evaluation of the data artefacts, measurement methodology, and
> fitness-for-purpose of the workflow study implemented in this repository.

---

## 1. Data Creation

### 1.1 Prompt Pack Design

The entire evaluation dataset is a single YAML file
(`prompts/workflow_prompts.yaml`) containing **16 hand-crafted prompts** in four
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
- **No ground-truth or reference outputs.** There is no baseline-expected
  answer stored alongside each prompt. Without references, there is no way to
  automate pass/fail scoring, compute BLEU/ROUGE, or detect regressions in
  CI. Every evaluation requires a human reading `examples.md`.
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

**Concerns:**

1. **No warmup pass.** The first prompt of each policy pays JIT compilation and
   CUDA kernel launch costs. This inflates the first measurement and distorts
   per-category averages. A single untimed warmup generation before the timed
   loop would eliminate this confound.
   (`study.py:56` — the prompt loop starts immediately after `prepare_model`.)

2. **Single run per prompt.** Each prompt-policy pair is evaluated exactly once.
   There is no repetition, no mean/standard-deviation, and no confidence
   interval. Even with deterministic decoding, latency and VRAM measurements
   have run-to-run variance from GPU scheduling, thermal throttling, and OS
   background activity. A minimum of 3–5 runs per pair is standard practice
   for hardware benchmarking.

3. **Fixed policy execution order.** Policies are evaluated in config-file order
   (baseline → safe → aggressive; `study.py:35`). Despite model reloading and
   `torch.cuda.empty_cache()` between policies (`study.py:73–78`), ordering
   effects can still arise from GPU thermal state, fragmented CUDA memory
   pools, and NCCL state. Shuffling the policy order (or running each policy
   in a fresh process) would strengthen the design.

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

### 3.2 What Is Missing from Evaluation

- **No automated quality scoring.** The `watch_for` field is a label for humans,
  not a scoring function. There is no code anywhere in the repository that
  compares model outputs to expected answers, computes similarity metrics, or
  flags regressions. The green/yellow/red decision framework described in
  `docs/study-scope.md:30–34` has **no programmatic implementation**.

- **No diff or delta computation.** When comparing baseline vs. compressed
  output, the reviewer must read both outputs in `examples.md` and mentally
  diff them. There is no automated text diff, no semantic similarity score, no
  output-length delta column in the CSV.

- **No pass/fail thresholds.** There are no configurable thresholds for latency
  regression, VRAM increase, or quality degradation that would let the harness
  flag a policy as "red" automatically.

- **No code-execution tests for coding prompts.** The four coding prompts ask
  for Python functions. The harness could execute the generated code and test
  it against known inputs — this would provide a strong, objective quality
  signal. Currently, correctness is assessed only by reading the output.

- **No numerical-answer checking for math prompts.** The four math prompts have
  deterministic correct answers (e.g. CAGR ≈ 20.5%, weighted average = 86.2,
  token estimate = 2040). Automated extraction and comparison of the final
  numeric answer would be straightforward and high-value.

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
| "Which policy is usable" | Partially | The harness collects latency, VRAM, and throughput metrics that help answer performance questions. But "usable" also means "output quality is acceptable", and there is no automated quality assessment. |
| "What degrades"         | Partially | Side-by-side markdown comparison shows *what changed*, but only to a human reader. There is no automated degradation detection. |
| "When I push harder"    | Weak | The prompt set does not vary in difficulty or length. There are no long-context, multi-turn, or adversarial prompts that would stress-test compression at its limits. |

**Overall pertinence: moderate.** The harness is well-architected and produces
the right *kinds* of data, but the evaluation methodology stops short of
answering the core question without significant manual effort. It is a strong
data-collection framework that lacks an analysis layer.

---

## 6. Recommendations

Ranked by impact-to-effort ratio:

### High Impact, Low Effort

1. **Add a warmup generation** before the timed prompt loop in `study.py`.
   A single untimed generation with a short prompt eliminates JIT/compilation
   noise from the first real measurement.

2. **Add reference answers for math prompts.** Store the expected numeric answer
   in each `PromptSpec` and compare against extracted values from the output.
   Four prompts, four numbers — immediate automated quality signal.

3. **Add output-length delta to CSV.** In `reporting.py`, include a column
   showing the percentage change in `output_tokens` relative to the baseline
   policy. Large length changes are a reliable early signal of compression
   damage.

### High Impact, Moderate Effort

4. **Add 8–12 long-context prompts** (1000–4000 tokens of input context).
   These should exercise the regime where KV-cache compression has the largest
   effect. Retrieval-augmented prompts with large context windows are ideal.

5. **Execute generated code for coding prompts.** Run the Python output through
   `exec()` or a subprocess with known test inputs. Report pass/fail in the
   CSV.

6. **Add 3–5 repetitions per prompt-policy pair** for latency/VRAM and report
   mean ± std. Keep only one text output per pair (they are deterministic).

### Medium Impact, Higher Effort

7. **Implement automated semantic comparison.** Use a lightweight similarity
   metric (e.g., sentence-transformer cosine similarity between baseline and
   compressed outputs) to produce a per-prompt quality score.

8. **Implement the green/yellow/red threshold system.** Define configurable
   thresholds in the study config (e.g., latency regression < 10% = green,
   10–25% = yellow, > 25% = red) and add a `verdict` column to the CSV.

9. **Randomize or interleave policy execution order.** Either shuffle policy
   order or run each policy in a separate subprocess to eliminate ordering
   confounds.

10. **Add multi-turn prompt sequences.** At least 2–3 multi-turn conversations
    to test whether compression artefacts accumulate across turns.

---

## Summary

| Dimension              | Rating       | Notes |
|------------------------|--------------|-------|
| Prompt relevance       | Good         | Domain-appropriate, workflow-specific prompts |
| Prompt coverage        | Insufficient | Too few, too short, no long-context stress tests |
| Measurement quality    | Good         | Correct timing, VRAM tracking, deterministic config |
| Statistical rigour     | Weak         | Single run, no warmup, fixed ordering |
| Automated evaluation   | Absent       | No scoring, no thresholds, no regression detection |
| Reporting              | Good         | Multiple formats, clean structure |
| Fitness for purpose    | Moderate     | Strong collection framework, but requires manual analysis to draw conclusions |

The harness is a solid foundation — the architecture is clean, the adapter
interface is well-designed, and the measurement instrumentation is correct. The
primary gaps are in **data coverage** (more and longer prompts) and in the
**evaluation layer** (automated quality scoring and threshold-based verdicts).
Addressing recommendations 1–3 would significantly strengthen the workflow with
minimal effort.
