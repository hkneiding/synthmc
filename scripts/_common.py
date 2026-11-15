from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make_progress_printer(t0: float):
    """A progress_callback printing "[stage] n_seen/n_total [elapsed, rate]"."""

    def progress(stage, n_seen, n_total, corpus):
        elapsed = time.time() - t0
        rate = n_seen / elapsed if elapsed > 0 else 0
        print(f"  [{stage}] {n_seen:>7d}/{n_total} [{elapsed:.0f}s, {rate:.0f}/s]", flush=True)

    return progress


__all__ = ["ROOT", "make_progress_printer"]
