from __future__ import annotations

import pickle
from pathlib import Path


class PicklePersistable:
    """Mixin providing `.save(path)` / `.load(path)` via plain pickling."""

    def save(self, path: str | Path) -> None:
        """Pickle this object to `path`."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path):
        """Unpickle from `path`, raising if it isn't an instance of `cls`."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return obj


__all__ = ["PicklePersistable"]
