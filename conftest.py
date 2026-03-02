import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DOCWAIN_FT_SRC = ROOT_DIR / "docwain-gptoss-finetune-pipeline" / "src"
if DOCWAIN_FT_SRC.exists() and str(DOCWAIN_FT_SRC) not in sys.path:
    sys.path.insert(0, str(DOCWAIN_FT_SRC))

import importlib

if "src" in sys.modules:
    del sys.modules["src"]
importlib.invalidate_caches()
import src  # noqa: F401
