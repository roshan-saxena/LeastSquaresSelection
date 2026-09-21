"""Locations for saved results, selected reruns and generated outputs."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("MLWA_RESULTS_DIR", ROOT / "results")).resolve()
OUTPUT = Path(os.environ.get("MLWA_OUTPUT_DIR", ROOT / "results/generated")).resolve()
