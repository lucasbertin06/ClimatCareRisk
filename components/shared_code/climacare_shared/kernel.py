"""Locate and load the compiled SmokeTransport kernel.

Inside the Tesseract image the extension module is copied next to
``tesseract_api.py`` in ``/tesseract``. On a development host it lives in the
CMake build tree of the component. Both cases are resolved here so that the
container glue and the test-suite share one import path.
"""

from __future__ import annotations

import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

__all__ = ["KernelNotBuiltError", "load_smoke_kernel", "smoke_kernel_search_paths"]

_MODULE_NAME = "smoke_kernel_cpp"
_ENV_VAR = "CLIMACARE_SMOKE_KERNEL_DIR"
_COMPONENT_RELATIVE = Path("components/tesseracts/smoke_transport_cpp/build")


class KernelNotBuiltError(ImportError):
    """Raised when the compiled C++ kernel cannot be located."""


def smoke_kernel_search_paths() -> list[Path]:
    """Return the candidate directories holding the compiled extension."""
    candidates: list[Path] = []
    override = os.environ.get(_ENV_VAR)
    if override:
        candidates.append(Path(override))
    # Inside the Tesseract image the module sits next to tesseract_api.py.
    candidates.append(Path("/tesseract"))
    # Development checkout: walk up from this file to the repository root.
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / _COMPONENT_RELATIVE
        if candidate.is_dir():
            candidates.append(candidate)
            break
    candidates.append(Path.cwd() / _COMPONENT_RELATIVE)
    return [path for path in candidates if path.is_dir()]


@lru_cache(maxsize=1)
def load_smoke_kernel() -> ModuleType:
    """Import the compiled kernel, extending ``sys.path`` if needed.

    Returns:
        The imported ``smoke_kernel_cpp`` extension module.

    Raises:
        KernelNotBuiltError: if the extension is not importable from any
            candidate directory.
    """
    try:
        return importlib.import_module(_MODULE_NAME)
    except ImportError:
        pass

    searched = smoke_kernel_search_paths()
    for directory in searched:
        entry = str(directory)
        if entry not in sys.path:
            sys.path.insert(0, entry)
        try:
            return importlib.import_module(_MODULE_NAME)
        except ImportError:
            continue

    raise KernelNotBuiltError(
        f"compiled kernel {_MODULE_NAME!r} not found. Build it with "
        "`make smoke-kernel` or set "
        f"{_ENV_VAR} to the directory holding the extension. Searched: "
        + ", ".join(str(path) for path in searched)
    )
