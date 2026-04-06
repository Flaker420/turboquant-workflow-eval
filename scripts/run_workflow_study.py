"""Thin shim for ``python -m turboquant_workflow_eval``.

The full CLI lives in ``turboquant_workflow_eval.__main__``. This script
exists only so users can still invoke ``python scripts/run_workflow_study.py
--study configs/studies/default_qwen35_9b.py ...``; everything is forwarded
unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from turboquant_workflow_eval.__main__ import main

if __name__ == "__main__":
    main()
