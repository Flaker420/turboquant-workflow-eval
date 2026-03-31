#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/bootstrap_runpod.sh [--download-model] [--tokenizer-only] [--model-config PATH]

Options:
  --download-model   Warm the Hugging Face cache after installing dependencies.
  --tokenizer-only   Only valid with --download-model. Cache tokenizer/config only.
  --model-config     Model config to use for cache warmup.
  -h, --help         Show this help message.

Environment overrides:
  WORKSPACE_ROOT     Default: /workspace
  VENV_DIR           Default: /workspace/venvs/qwen35-turboquant-study
  CACHE_ROOT         Default: /workspace/.cache/huggingface
  OUTPUT_ROOT        Default: /workspace/outputs
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DOWNLOAD_MODEL=0
TOKENIZER_ONLY=0
MODEL_CONFIG="configs/model/qwen35_9b_text_only.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --download-model)
      DOWNLOAD_MODEL=1
      shift
      ;;
    --tokenizer-only)
      TOKENIZER_ONLY=1
      shift
      ;;
    --model-config)
      MODEL_CONFIG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$TOKENIZER_ONLY" -eq 1 && "$DOWNLOAD_MODEL" -ne 1 ]]; then
  echo "--tokenizer-only requires --download-model" >&2
  exit 2
fi

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
VENV_DIR="${VENV_DIR:-${WORKSPACE_ROOT}/venvs/qwen35-turboquant-study}"
CACHE_ROOT="${CACHE_ROOT:-${WORKSPACE_ROOT}/.cache/huggingface}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_ROOT}/outputs}"

mkdir -p "$(dirname "${VENV_DIR}")" "${CACHE_ROOT}" "${OUTPUT_ROOT}"

python -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${REPO_DIR}/requirements.txt"

cat > "${REPO_DIR}/.env.runpod" <<EOF
export HF_HOME="${CACHE_ROOT}"
export TRANSFORMERS_CACHE="${CACHE_ROOT}"
export PYTHONPATH="${REPO_DIR}/src"
export TOKENIZERS_PARALLELISM=false
EOF

# shellcheck disable=SC1091
source "${REPO_DIR}/.env.runpod"

if [[ "$DOWNLOAD_MODEL" -eq 1 ]]; then
  DOWNLOAD_ARGS=(--model-config "$MODEL_CONFIG")
  if [[ "$TOKENIZER_ONLY" -eq 1 ]]; then
    DOWNLOAD_ARGS+=(--tokenizer-only)
  fi
  python "${REPO_DIR}/scripts/download_model.py" "${DOWNLOAD_ARGS[@]}"
fi

cat <<EOF
Bootstrap complete.

Next steps:
  source "${REPO_DIR}/.env.runpod"
  source "${VENV_DIR}/bin/activate"
  cd "${REPO_DIR}"
  python scripts/validate_environment.py
  pytest -q

Optional cache warmup during bootstrap:
  bash scripts/bootstrap_runpod.sh --download-model
EOF
