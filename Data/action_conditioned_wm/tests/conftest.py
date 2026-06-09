"""Pytest bootstrap: put the package dir and the Data/ dir on sys.path so flat
imports (``from wm_simulator import ...``) and sibling reuse (``from graph_utils
import ...``, ``from datasets.jazz import ...``) both resolve."""

import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent  # Data/action_conditioned_wm
DATA_DIR = PKG_DIR.parent  # Data

for _p in (str(PKG_DIR), str(DATA_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
