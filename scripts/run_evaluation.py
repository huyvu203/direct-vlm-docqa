#!/usr/bin/env python3
"""Command-line entry point for prediction evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_vlm.evaluation import app


if __name__ == "__main__":
    app()
