"""Configuration and entry-point layer for the dp-code thesis pipeline.

`dpcode` sits *in front of* `project/` and `tools/`; it does not replace them.
CLAM (``project/CLAM/main.py``) keeps its argparse surface and is dispatched as a
subprocess, so no numerical behaviour can drift through this package.

Importing this module is deliberately cheap: only :mod:`dpcode.paths` (stdlib only)
is pulled in eagerly, so ``import dpcode; dpcode.paths.repo_root()`` costs a few
``stat`` calls and no OmegaConf/Hydra/torch import.
"""

from . import paths

__all__ = ["paths", "__version__"]

__version__ = "0.1.0"
