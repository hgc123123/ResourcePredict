#!/usr/bin/env python3
"""Export evaluation tables and figures from computed results.

No model fitting, policy selection, or historical result substitution occurs.
The help and dry-run paths need only Python's standard library.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from publication_reporting.report import main

if __name__ == "__main__":
    raise SystemExit(main())
