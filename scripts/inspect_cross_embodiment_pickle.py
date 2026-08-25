#!/usr/bin/env python3
"""Safely inspect RoboTwin trajectory pickle structure without arbitrary globals."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np


ALLOWED_GLOBALS = {
    ("numpy", "dtype"): np.dtype,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        try:
            return ALLOWED_GLOBALS[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(f"forbidden global: {module}.{name}") from exc


def load(path: Path) -> Any:
    with path.open("rb") as handle:
        return RestrictedUnpickler(handle).load()


def describe(value: Any, indent: str = "", depth: int = 0) -> None:
    if isinstance(value, dict):
        print(f"{indent}dict keys={list(value)}")
        if depth < 3:
            for key, child in value.items():
                print(f"{indent}  [{key!r}]")
                describe(child, indent + "    ", depth + 1)
    elif isinstance(value, (list, tuple)):
        print(f"{indent}{type(value).__name__} len={len(value)}")
        if value and depth < 3:
            describe(value[0], indent + "  first: ", depth + 1)
            if len(value) > 1:
                describe(value[-1], indent + "  last:  ", depth + 1)
    elif isinstance(value, np.ndarray):
        print(f"{indent}ndarray shape={value.shape} dtype={value.dtype}")
    else:
        print(f"{indent}{type(value).__name__}: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        print(f"FILE {path}")
        describe(load(path))


if __name__ == "__main__":
    main()
