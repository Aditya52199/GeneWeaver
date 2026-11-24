"""
GeneWeaver Dashboard Package.

Provides a Textual Terminal User Interface for tracking genomic chunking,
real-time progress bars, and sequence alignment.
"""

from typing import Any

from dashboard.layout import MainLayout
from dashboard.progress_view import ProgressView
from dashboard.widgets import ChunkStreamTable, StatCard


def run_dashboard() -> None:
    """Launch the GeneWeaver Textual Interface."""
    from dashboard.app import run_dashboard as _run
    _run()


def __getattr__(name: str) -> Any:
    """Lazy-import GeneWeaverApp to prevent runpy circular import warnings."""
    if name == "GeneWeaverApp":
        from dashboard.app import GeneWeaverApp
        return GeneWeaverApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChunkStreamTable",
    "GeneWeaverApp",
    "MainLayout",
    "ProgressView",
    "StatCard",
    "run_dashboard",
]
