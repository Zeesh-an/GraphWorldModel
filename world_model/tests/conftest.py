import sys
from pathlib import Path

# Put world_model/ and the repo root (for `import data...`) on the path
_WM = Path(__file__).resolve().parent.parent
_ROOT = _WM.parent
for _p in (str(_WM), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
