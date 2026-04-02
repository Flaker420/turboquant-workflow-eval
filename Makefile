PYTHONPATH := src
export PYTHONPATH

MODEL_CONFIG ?= configs/model/qwen35_9b_text_only.yaml
EXPERIMENT_CONFIG ?= configs/experiments/preflight_stats.yaml
STUDY_CONFIG ?= configs/studies/default.yaml
POLICY_CONFIGS ?= configs/policies/baseline.yaml
OUTPUT_DIR ?= outputs/study_run
DOWNLOAD_OUTPUT ?= outputs/download_summary.json
BACKEND_STUDY_CONFIG ?= configs/backend_studies/default.yaml
BACKEND_CONFIGS ?= configs/backends/qwen35_openai_server.yaml
BACKEND_PROBE_PROMPT ?= Say hello in one short sentence.

.PHONY: help test validate download-model list-attention preflight study backend-probe backend-study

help:
	@printf "%s\n" \
	  "make validate                    # Import check + CUDA visibility" \
	  "make test                        # Run unit tests" \
	  "make download-model             # Warm HF cache using MODEL_CONFIG" \
	  "make list-attention             # Discover attention blocks" \
	  "make preflight                  # Run preflight instrumentation" \
	  "make study POLICY_CONFIGS=...   # Run workflow study" \
  "make backend-probe             # Probe OpenAI-compatible backend" \
  "make backend-study             # Run backend workflow study"

test:
	pytest -q

validate:
	python scripts/validate_environment.py

download-model:
	python scripts/download_model.py --model-config $(MODEL_CONFIG) --output $(DOWNLOAD_OUTPUT)

list-attention:
	python scripts/list_attention_blocks.py --model-config $(MODEL_CONFIG)

preflight:
	python scripts/run_preflight_stats.py --experiment-config $(EXPERIMENT_CONFIG) --output-dir outputs/preflight_smoke

study:
	python scripts/run_workflow_study.py --study-config $(STUDY_CONFIG) --policy-configs $(POLICY_CONFIGS) --output-dir $(OUTPUT_DIR)

backend-probe:
	python scripts/probe_openai_backend.py --backend-config $(BACKEND_CONFIGS) --prompt "$(BACKEND_PROBE_PROMPT)"

backend-study:
	python scripts/run_backend_study.py --study-config $(BACKEND_STUDY_CONFIG) --backend-configs $(BACKEND_CONFIGS) --output-dir $(OUTPUT_DIR)
