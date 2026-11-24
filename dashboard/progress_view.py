"""
Genomic Chunking Progress View for GeneWeaver Textual Interface.

Tracks the genomic chunking process with live progress bars,
real-time throughput metrics, and dynamic chunk stream tables.
"""

from typing import Optional
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Button, Input, Label, ProgressBar, Static

from dashboard.widgets import ChunkStreamTable, StatCard


class ProgressView(Widget):
    """Container view for genomic ingestion and chunking progress tracking."""

    def compose(self) -> ComposeResult:
        with Vertical(classes="view-container"):
            # 1. Top Metrics Cards
            with Horizontal(classes="stats-grid"):
                yield StatCard("TOTAL NUCLEOTIDES", "0 bp", "0.0% of genome", id="stat-total-bp")
                yield StatCard("CHUNKS CREATED", "0", "0 KB in VRAM", id="stat-chunks")
                yield StatCard("THROUGHPUT", "0 bp/s", "0.00 chunks/s", id="stat-throughput")
                yield StatCard("ACTIVE CHROMOSOME", "-", "Standing by", id="stat-active-chr")

            # 2. Dual Progress Bars (Genome Total + Active Chromosome)
            with Vertical(classes="progress-container"):
                with Horizontal(classes="progress-header"):
                    yield Label("Genome Ingestion Progress", classes="progress-title", id="lbl-genome-progress")
                    yield Label("0 / 0 bp (0.0%) | ETA: --:--", classes="progress-detail", id="lbl-genome-detail")
                yield ProgressBar(id="genome-progress", total=100, show_eta=False)

                with Horizontal(classes="progress-header"):
                    yield Label("Active Chromosome Progress", classes="progress-title", id="lbl-chr-progress")
                    yield Label("0 / 0 bp (0.0%)", classes="progress-detail", id="lbl-chr-detail")
                yield ProgressBar(id="chr-progress", total=100, show_eta=False)

            # 3. Interactive Pipeline Controls
            with Horizontal(classes="controls-box"):
                yield Input(
                    value="data/mock_human_genome.fasta",
                    placeholder="Reference FASTA path...",
                    classes="control-input",
                    id="input-fasta",
                )
                yield Input(
                    value="10000",
                    placeholder="Chunk size (bp)",
                    classes="control-input-sm",
                    id="input-chunk-size",
                )
                yield Input(
                    value="100",
                    placeholder="Overlap (bp)",
                    classes="control-input-sm",
                    id="input-overlap",
                )
                yield Button("▶ Start Chunking", classes="btn-start", id="btn-start")
                yield Button("⏸ Pause", classes="btn-pause", id="btn-pause")
                yield Button("↺ Reset", classes="btn-reset", id="btn-reset")

            # 4. Live Chunk Stream Table
            with Container(classes="table-container"):
                yield ChunkStreamTable(id="chunk-table")
