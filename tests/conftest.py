import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Run the suite single-threaded. These tests are hundreds of operations on tiny
# tensors -- 16x16 panels, 6x6 windows -- and for work that small the threading
# overhead dwarfs the arithmetic: torch fans each operation out across every
# core, then spends longer synchronising than computing.
#
# Measured on this suite: 313 s across 18 cores at 1773% CPU, against 30 s on
# one core at 99%. A factor of 10.4, by using LESS of the machine.
#
# The environment variables must be set before torch imports to take effect, so
# they live here rather than in pytest.ini, which has no way to set them.
# `set_num_threads` covers the case where torch was already imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch  # noqa: E402  (must follow the environment variables above)

torch.set_num_threads(1)
