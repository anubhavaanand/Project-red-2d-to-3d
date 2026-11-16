"""One loader for hand-picked / object-id list files.

DATA-01: the same handpicked.txt was parsed three different ways:

* gso1000/select_1000.py skipped blank lines only, so a # comment line
  became an object ID.
* src/evaluation/benchmark_runner.py checked line.startswith("#") on the
  *unstripped* line, so an indented comment slipped through.
* the Kaggle notebooks skipped comments properly.

A # line was therefore an object ID in one path and a comment in another.
This module is the single implementation.
"""

from __future__ import annotations

from pathlib import Path


def load_object_ids(path: str | Path) -> list[str]:
    """Read a hand-picked object list: one ID per line, # comments ignored.

    Blank lines and lines whose first non-space character is # are skipped.
    Order is preserved and duplicates are removed (first occurrence wins).
    """
    text = Path(path).read_text(encoding="utf-8")
    ids: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in seen:
            continue
        seen.add(line)
        ids.append(line)
    return ids
