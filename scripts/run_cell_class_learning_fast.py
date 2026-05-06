"""Compatibility wrapper for the production cell-class learning command."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from scripts.run_cell_class_learning import main  # noqa: E402


if __name__ == "__main__":
    main()
