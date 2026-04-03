PYTHONPATH := src
export PYTHONPATH

MODEL_CONFIG ?= configs/model/qwen35_9b_text_only.yaml
EXPERIMENT_CONFIG ?= configs/experiments/preflight_stats.yaml
STUDY_CONFIG ?= configs/studies/default.yaml
POLICY_CONFIGS ?= configs/policies/baseline.yaml
OUTPUT_DIR ?= outputs/study_run
DOWNLOAD_OUTPUT ?= outputs/download_summary.json

.PHONY: help test validate download-model download-all check-cache list-attention preflight study

help:
	@printf "%s\n" \
	  "make validate                    # Import check + CUDA visibility" \
	  "make test                        # Run unit tests" \
	  "make download-model             # Warm HF cache using MODEL_CONFIG" \
	  "make download-all               # Download all models in configs/model/" \
	  "make check-cache                # Check HF cache status for all models" \
	  "make list-attention             # Discover attention blocks" \
	  "make preflight                  # Run preflight instrumentation" \
	  "make study POLICY_CONFIGS=...   # Run workflow study"

test:
	pytest -q

validate:
	python scripts/validate_environment.py

download-model:
	python scripts/download_model.py --model-config $(MODEL_CONFIG) --output $(DOWNLOAD_OUTPUT)

download-all:
	python scripts/download_model.py --all --output $(DOWNLOAD_OUTPUT)

check-cache:
	python scripts/download_model.py --check-only

list-attention:
	python scripts/list_attention_blocks.py --model-config $(MODEL_CONFIG)

preflight:
	python scripts/run_preflight_stats.py --experiment-config $(EXPERIMENT_CONFIG) --output-dir outputs/preflight_smoke

study:
	python scripts/run_workflow_study.py --study-config $(STUDY_CONFIG) --policy-configs $(POLICY_CONFIGS) --output-dir $(OUTPUT_DIR)
