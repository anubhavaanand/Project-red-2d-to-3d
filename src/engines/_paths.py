"""Safe discovery of vendored third-party model repositories.

The engine wrappers import code from repos that are cloned next to this
project or into a scratch directory at runtime. Those directories are added
to ``sys.path`` so ``import tsr`` / ``import trellis`` / ``import spar3d``
resolve.

Two hazards with the previous approach:

* ``sys.path.insert(0, ...)`` puts the scratch directory *ahead* of the
  standard library and site-packages, so any same-named module in that
  directory — planted or accidental — shadows the real one.
* Directories such as ``/tmp`` are world-writable, so another local user can
  drop a file there.

Appending instead of prepending keeps installed dependencies authoritative
while still making the vendored package importable. Callers get a bool so they
can raise a clear error when the repo was never found.
"""

from __future__ import annotations

import sys
from pathlib import Path


def extend_sys_path(*candidates: Path) -> bool:
    """Append the first existing candidate directory to ``sys.path``.

    Appending (rather than inserting at position 0) means anything already
    installed — the standard library, site-packages, real dependencies — keeps
    priority, so a file in a scratch directory cannot shadow it.

    Returns ``True`` when a directory was added or was already present.
    """
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        entry = str(candidate)
        if entry not in sys.path:
            sys.path.append(entry)
        return True
    return False