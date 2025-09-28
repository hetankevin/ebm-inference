"""Expose a patched ``InferableEBMRegressor`` pulled from local sources.

If the repository contains patched copies of ``_ebm.py`` or ``_utils.py`` in
``src/``, they are loaded in place of the corresponding modules shipped with the
installed :mod:`interpret` package.  Otherwise, the standard package modules are
used.  This lets users simply ``pip install interpret`` yet still pick up local
fixes without having to rebuild wheels.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent
_INTERPRET_CORE = _ROOT / "interpret" / "python" / "interpret-core"
_SRC_DIR = _ROOT / "src"


def _maybe_prepend(path: Path) -> None:
    if path.is_dir():
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


# Prefer a checked-out interpret-core if present.
_maybe_prepend(_INTERPRET_CORE)


def _load_override(fullname: str, source: Path) -> None:
    """Load ``source`` under the fully-qualified ``fullname`` module name."""

    if not source.is_file():  # no override available
        return

    spec = importlib.util.spec_from_file_location(fullname, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create spec for {fullname} from {source}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module  # ensure relatives resolve to this module
    spec.loader.exec_module(module)

    parent_name = fullname.rsplit(".", 1)[0]
    parent = importlib.import_module(parent_name)
    setattr(parent, fullname.split(".")[-1], module)


# Ensure the base interpret package is importable before we apply overrides.
try:
    importlib.import_module("interpret.glassbox._ebm")
except ModuleNotFoundError as exc:  # pragma: no cover - diagnostic guard
    raise ModuleNotFoundError(
        "Unable to import interpret.glassbox._ebm. Install the 'interpret' package first."
    ) from exc


_OVERRIDES = {
    "interpret.glassbox._ebm._utils": _SRC_DIR / "_utils.py",
    "interpret.glassbox._ebm._ebm": _SRC_DIR / "_ebm.py",
}

# Apply overrides in dependency order so relative imports resolve correctly.
for module_name in ("interpret.glassbox._ebm._utils", "interpret.glassbox._ebm._ebm"):
    # Remove any previously loaded version before we inject our replacement.
    sys.modules.pop(module_name, None)
    _load_override(module_name, _OVERRIDES[module_name])


# If no local override exists, fall back to the package implementation.
_ebm_module = importlib.import_module("interpret.glassbox._ebm._ebm")
_ebm_pkg = importlib.import_module("interpret.glassbox._ebm")

if hasattr(_ebm_module, "__all__"):
    for _name in _ebm_module.__all__:
        setattr(_ebm_pkg, _name, getattr(_ebm_module, _name))

InferableEBMRegressor = getattr(_ebm_module, "InferableEBMRegressor")
