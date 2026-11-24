"""
Custom Textual UI Widgets for GeneWeaver Dashboard.
"""

from typing import Optional
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import DataTable, Label, Static


class StatCard(Widget):
    """A clean metric card widget with title, large numerical value, and subtext."""

    DEFAULT_CSS = """
    StatCard {
        background: #162032;
        border: round #2a3b5c;
        padding: 0 1;
        height: 100%;
        content-align: center middle;
    }
    """

    def __init__(
        self,
        title: str,
        value: str = "-",
        subtext: str = "",
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ):
        super().__init__(id=id, classes=classes)
        self._title = title
        self._value = value
        self._subtext = subtext

    def compose(self) -> ComposeResult:
        yield Label(self._title, classes="stat-title", id=f"{self.id}-title" if self.id else None)
        yield Label(self._value, classes="stat-value", id=f"{self.id}-val" if self.id else None)
        yield Label(self._subtext, classes="stat-sub", id=f"{self.id}-sub" if self.id else None)

    def update_values(self, value: str, subtext: Optional[str] = None) -> None:
        """Update displayed stat value and subtext."""
        self._value = value
        if subtext is not None:
            self._subtext = subtext

        val_label = self.query_one(f"#{self.id}-val", Label) if self.id else None
        if val_label:
            val_label.update(self._value)

        if subtext is not None:
            sub_label = self.query_one(f"#{self.id}-sub", Label) if self.id else None
            if sub_label:
                sub_label.update(self._subtext)


class ChunkStreamTable(DataTable):
    """DataTable displaying real-time stream of genomic chunks."""

    def on_mount(self) -> None:
        """Configure table columns with rich styling."""
        self.cursor_type = "row"
        self.zebra_stripes = True
        self.add_column("Chunk ID", width=12)
        self.add_column("Chromosome", width=14)
        self.add_column("Genomic Range", width=22)
        self.add_column("Length (bp)", width=14)
        self.add_column("GC %", width=10)
        self.add_column("N Bases", width=10)
        self.add_column("VRAM Array", width=14)
        self.add_column("Status", width=12)

    def append_chunk(
        self,
        chunk_id: int,
        chromosome: str,
        start_pos: int,
        end_pos: int,
        length: int,
        gc_content: float,
        ambiguous_count: int,
        status: str = "Ready",
    ) -> None:
        """Append a newly processed chunk to the data table."""
        # Style status with color
        status_text = Text(status)
        if status == "Ready":
            status_text.stylize("bold green")
        elif status == "Processing":
            status_text.stylize("bold yellow")
        else:
            status_text.stylize("bold cyan")

        gc_text = Text(f"{gc_content:.1f}%")
        if gc_content < 35 or gc_content > 65:
            gc_text.stylize("yellow")
        else:
            gc_text.stylize("cyan")

        self.add_row(
            f"#{chunk_id:04d}",
            chromosome,
            f"[{start_pos:,} - {end_pos:,}]",
            f"{length:,}",
            gc_text,
            f"{ambiguous_count:,}",
            f"{length / 1024:.1f} KB",
            status_text,
        )
