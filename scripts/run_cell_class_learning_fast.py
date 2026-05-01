"""Compatibility wrapper for the production cell-class learning command.

The only supported cell-learning condition stack is the canonical Figure 7
Rucci/Boi early/late cycle. This wrapper exists for old shell history; it
delegates to ``scripts/run_cell_class_learning.py`` and exposes no alternate
movement-spectrum path.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from scripts.run_cell_class_learning import main  # noqa: E402


if __name__ == "__main__":
    main()
