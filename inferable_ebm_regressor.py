"""Helper module to load the locally patched InferableEBMRegressor.

This mirrors the module layout expected by the experiments scripts and
ensures our checked-out InterpretML sources are used ahead of any
site-packages installation.
"""
from __future__ import annotations

import importlib
import os
import sys

# Path to the local interpret-core package within this repository.
_ROOT = os.path.dirname(__file__)
_INTERPRET_CORE = os.path.join(_ROOT, "interpret", "python", "interpret-core")

# Prepend the checkout to sys.path so python resolves to our patched code.
if os.path.isdir(_INTERPRET_CORE) and _INTERPRET_CORE not in sys.path:
    sys.path.insert(0, _INTERPRET_CORE)

try:
    # Import via the canonical package path so relative imports inside the
    # module continue to work.
    _ebm_module = importlib.import_module("interpret.glassbox._ebm._ebm")
except ModuleNotFoundError as exc:  # pragma: no cover - diagnostic guard
    raise ModuleNotFoundError(
        "Could not import interpret.glassbox._ebm._ebm from the local checkout. "
        "Ensure you have the InterpretML sources under interpret/python/interpret-core"
    ) from exc

# Re-export the estimator so experiments can simply import it from here.
InferableEBMRegressor = getattr(_ebm_module, "InferableEBMRegressor")
