#!/usr/bin/env python3
"""Compatibility alias for the explicitly synthetic CPU pipeline smoke test."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.smoke_reviewer_pipeline import main

if __name__ == "__main__":
    main()
