#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/bootstrap_runpod.sh [--download-model] [--tokenizer-only] [--no-fast-path] [--model-config PATH]

Options:
  --download-model   Warm the Hugging Face cache for the configured model.
  --download-all     Download all models found in configs/model/.
  --tokenizer-only   Only valid with --download-model/--download-all. Cache tokenizer/config only.
  --fast-path        No-op, kept for backwards compatibility. Fast-path packages are now installed by default.
  --no-fast-path     Skip fast-path package installation (flash-linear-attention, causal-conv1d).
  --model-config     Model config to use for single-model cache warmup.
  -h, --help         Show this help message.

Environment overrides:
  WORKSPACE_ROOT     Default: /workspace
  VENV_DIR           Default: /workspace/venvs/turboquant-eval
  CACHE_ROOT         Default: /workspace/.cache/huggingface
  OUTPUT_ROOT        Default: /workspace/outputs
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DOWNLOAD_MODEL=0
DOWNLOAD_ALL=0
TOKENIZER_ONLY=0
SKIP_FAST_PATH=0
MODEL_CONFIG="configs/model/qwen35_9b_text_only.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --download-model)
      DOWNLOAD_MODEL=1
      shift
      ;;
    --download-all)
      DOWNLOAD_ALL=1
      shift
      ;;
    --tokenizer-only)
      TOKENIZER_ONLY=1
      shift
      ;;
    --fast-path)
      # No-op, kept for backwards compatibility.
      shift
      ;;
    --no-fast-path)
      SKIP_FAST_PATH=1
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

if [[ "$TOKENIZER_ONLY" -eq 1 && "$DOWNLOAD_MODEL" -ne 1 && "$DOWNLOAD_ALL" -ne 1 ]]; then
  echo "--tokenizer-only requires --download-model or --download-all" >&2
  exit 2
fi

if [[ "$DOWNLOAD_MODEL" -eq 1 && "$DOWNLOAD_ALL" -eq 1 ]]; then
  echo "--download-model and --download-all are mutually exclusive" >&2
  exit 2
fi

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
VENV_DIR="${VENV_DIR:-${WORKSPACE_ROOT}/venvs/turboquant-eval}"
CACHE_ROOT="${CACHE_ROOT:-${WORKSPACE_ROOT}/.cache/huggingface}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_ROOT}/outputs}"

mkdir -p "$(dirname "${VENV_DIR}")" "${CACHE_ROOT}" "${OUTPUT_ROOT}"

python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip "setuptools<82" wheel ninja
python -m pip install -r "${REPO_DIR}/requirements-runpod-cu128.txt"
python -m pip install -r "${REPO_DIR}/requirements.txt"
python -m pip install -e "${REPO_DIR}[dev]"

if [[ "$SKIP_FAST_PATH" -ne 1 ]]; then
  echo "Attempting fast-path packages (flash-linear-attention, causal-conv1d)..."
  python -m pip install flash-linear-attention || echo "  flash-linear-attention: install failed (non-fatal)"
  python -m pip install causal-conv1d --no-build-isolation || echo "  causal-conv1d: install failed (non-fatal)"
fi

cat > "${REPO_DIR}/.env.runpod" <<EOF
export HF_HOME="${CACHE_ROOT}"
export TRANSFORMERS_CACHE="${CACHE_ROOT}"
export PYTHONPATH="${REPO_DIR}/src"
export TOKENIZERS_PARALLELISM=false
EOF

source "${REPO_DIR}/.env.runpod"

if [[ "$DOWNLOAD_ALL" -eq 1 ]]; then
  DOWNLOAD_ARGS=(--all)
  if [[ "$TOKENIZER_ONLY" -eq 1 ]]; then
    DOWNLOAD_ARGS+=(--tokenizer-only)
  fi
  python "${REPO_DIR}/scripts/download_model.py" "${DOWNLOAD_ARGS[@]}"
elif [[ "$DOWNLOAD_MODEL" -eq 1 ]]; then
  DOWNLOAD_ARGS=(--model-config "$MODEL_CONFIG")
  if [[ "$TOKENIZER_ONLY" -eq 1 ]]; then
    DOWNLOAD_ARGS+=(--tokenizer-only)
  fi
  python "${REPO_DIR}/scripts/download_model.py" "${DOWNLOAD_ARGS[@]}"
fi

cat <<EOF
Bootstrap complete.

Environment summary:
  VENV_DIR=${VENV_DIR}
  CACHE_ROOT=${CACHE_ROOT}
  OUTPUT_ROOT=${OUTPUT_ROOT}

Next steps:
  source "${REPO_DIR}/.env.runpod"
  source "${VENV_DIR}/bin/activate"
  cd "${REPO_DIR}"
  python scripts/validate_environment.py
  pytest -q
EOF
