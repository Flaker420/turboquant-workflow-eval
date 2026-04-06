PYTHONPATH := src
export PYTHONPATH

MODEL_CONFIG ?= configs/model/qwen35_9b_text_only.py
EXPERIMENT_CONFIG ?= configs/experiments/preflight_stats.py
STUDY_CONFIG ?= configs/studies/default_qwen35_9b.py
POLICY_CONFIGS ?= configs/policies/baseline.py
OUTPUT_DIR ?= outputs/study_run
DOWNLOAD_OUTPUT ?= outputs/download_summary.json
RESCORE_INPUT ?= $(OUTPUT_DIR)/rows.jsonl

.PHONY: help test validate download-model download-all check-cache list-attention preflight dry-run smoke-test study study-full rescore generate-prompts ui

help:
	@printf "%s\n" \
	  "make validate                    # Import check + CUDA visibility" \
	  "make test                        # Run unit tests" \
	  "make download-model             # Warm HF cache using MODEL_CONFIG" \
	  "make download-all               # Download all models in configs/model/" \
	  "make check-cache                # Check HF cache status for all models" \
	  "make list-attention             # Discover attention blocks" \
	  "make preflight                  # Run preflight instrumentation" \
	  "make dry-run                    # Validate configs without GPU (<1s)" \
	  "make smoke-test                 # Quick single-prompt test" \
	  "make study POLICY_CONFIGS=...   # Run workflow study" \
	  "make study-full POLICY_CONFIGS=... # Run study with full prompt set" \
	  "make rescore RESCORE_INPUT=...  # Re-score existing results (no GPU)" \
	  "make generate-prompts           # Generate long-context prompts" \
	  "make ui                         # Launch Gradio web UI"

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

dry-run:
	python -m turboquant_workflow_eval --study $(STUDY_CONFIG) --dry-run

smoke-test:
	python -m turboquant_workflow_eval --study $(STUDY_CONFIG) --single --output-dir $(OUTPUT_DIR)

study:
	python scripts/run_workflow_study.py --study $(STUDY_CONFIG) --policies $(POLICY_CONFIGS) --output-dir $(OUTPUT_DIR)

study-full:
	python scripts/run_workflow_study.py --study configs/studies/full.py --policies $(POLICY_CONFIGS) --output-dir $(OUTPUT_DIR)

rescore:
	python -m turboquant_workflow_eval --rescore $(RESCORE_INPUT) --output-dir $(OUTPUT_DIR)

generate-prompts:
	python scripts/generate_prompts.py --model-config $(MODEL_CONFIG) --output prompts/generated_long_context.yaml

ui:
	python app.py
