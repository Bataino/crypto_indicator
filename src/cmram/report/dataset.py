"""Dataset Report: sources, coverage, missingness, survivorship limits."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_dataset_report(
    output_path: Path | str,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write Dataset Report markdown/HTML.

    Raises:
        NotImplementedError: stub.
    """
    raise NotImplementedError("write_dataset_report stub")
