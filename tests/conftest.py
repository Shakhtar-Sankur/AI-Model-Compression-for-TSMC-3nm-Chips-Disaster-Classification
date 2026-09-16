"""Put the repository root on sys.path so the modules under test import.

They sit at the root rather than in an installed package, and bare `pytest` does
not add the working directory to sys.path — only `python -m pytest` does. Without
this, collection fails with ModuleNotFoundError on a clean checkout and on CI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
