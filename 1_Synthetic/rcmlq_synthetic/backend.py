"""Load the numerical core shared with the Robot experiment in this checkout."""
from pathlib import Path
import sys

BACKEND_PATH = Path(__file__).resolve().parents[2] / "2_Robot"
if not (BACKEND_PATH / "metaworld_fast" / "model.py").is_file():
    raise ImportError("Use an RC-MLQ checkout containing both experiment directories")
sys.path.insert(0, str(BACKEND_PATH))
