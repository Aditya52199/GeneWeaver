"""
GeneWeaver Textual Interface Application.

Provides a terminal UI with real-time progress bars tracking
the genomic chunking process, live throughput telemetry, and sequence alignment.
"""

import os
import sys
import time
from typing import Optional

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Button, Footer, Header, Input, Label, ProgressBar

from dashboard.layout import MainLayout
from dashboard.widgets import ChunkStreamTable, StatCard
from genomic_parser.fasta_reader import FastaReader
from genomic_parser.sequence_chunker import SequenceChunker


class GeneWeaverApp(App):
    """GeneWeaver Textual Interactive Terminal Dashboard."""

    TITLE = "GeneWeaver: GPU-Accelerated CRISPR Alignment Engine"
    SUB_TITLE = "Real-Time Genomic Chunking & Alignment Telemetry"
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("s", "start_chunking", "Start Chunking", show=True),
        Binding("p", "toggle_pause", "Pause/Resume", show=True),
        Binding("r", "reset_chunking", "Reset", show=True),
        Binding("d", "toggle_dark", "Toggle Dark", show=False),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chunking_active = False
        self.chunking_paused = False
        self.total_processed_bp = 0
        self.total_chunks_created = 0
        self.pipeline_start_time = 0.0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield MainLayout()
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button events."""
        if event.button.id == "btn-start":
            self.action_start_chunking()
        elif event.button.id == "btn-pause":
            self.action_toggle_pause()
        elif event.button.id == "btn-reset":
            self.action_reset_chunking()

    def action_start_chunking(self) -> None:
        """Start the genomic chunking pipeline worker."""
        if self.chunking_active:
            if self.chunking_paused:
                self.chunking_paused = False
                btn_pause = self.query_one("#btn-pause", Button)
                btn_pause.label = "⏸ Pause"
            return

        self.chunking_active = True
        self.chunking_paused = False
        self.pipeline_start_time = time.perf_counter()

        btn_start = self.query_one("#btn-start", Button)
        btn_start.disabled = True
        btn_start.label = "Processing..."

        # Launch background worker thread
        self.chunking_worker()

    def action_toggle_pause(self) -> None:
        """Pause or resume the chunking pipeline."""
        if not self.chunking_active:
            return

        self.chunking_paused = not self.chunking_paused
        btn_pause = self.query_one("#btn-pause", Button)
        btn_pause.label = "▶ Resume" if self.chunking_paused else "⏸ Pause"

    def action_reset_chunking(self) -> None:
        """Reset chunking state, progress bars, and table."""
        self.chunking_active = False
        self.chunking_paused = False
        self.total_processed_bp = 0
        self.total_chunks_created = 0

        # Reset buttons
        btn_start = self.query_one("#btn-start", Button)
        btn_start.disabled = False
        btn_start.label = "▶ Start Chunking"

        btn_pause = self.query_one("#btn-pause", Button)
        btn_pause.label = "⏸ Pause"

        # Reset Progress Bars
        g_prog = self.query_one("#genome-progress", ProgressBar)
        g_prog.update(total=100, progress=0)

        c_prog = self.query_one("#chr-progress", ProgressBar)
        c_prog.update(total=100, progress=0)

        # Reset Labels
        self.query_one("#lbl-genome-detail", Label).update("0 / 0 bp (0.0%) | ETA: --:--")
        self.query_one("#lbl-chr-detail", Label).update("0 / 0 bp (0.0%)")

        # Reset Stat Cards
        self.query_one("#stat-total-bp", StatCard).update_values("0 bp", "0.0% of genome")
        self.query_one("#stat-chunks", StatCard).update_values("0", "0 KB in VRAM")
        self.query_one("#stat-throughput", StatCard).update_values("0 bp/s", "0.00 chunks/s")
        self.query_one("#stat-active-chr", StatCard).update_values("-", "Standing by")

        # Clear data table
        table = self.query_one("#chunk-table", ChunkStreamTable)
        table.clear()

    @work(thread=True)
    def chunking_worker(self) -> None:
        """Background thread executing the BioPython chunking pipeline."""
        # Read parameters from UI
        fasta_path = self.query_one("#input-fasta", Input).value.strip()
        chunk_size_str = self.query_one("#input-chunk-size", Input).value.strip()
        overlap_str = self.query_one("#input-overlap", Input).value.strip()

        try:
            chunk_size = int(chunk_size_str)
            overlap = int(overlap_str)
        except ValueError:
            chunk_size = 10_000
            overlap = 100

        if not os.path.exists(fasta_path):
            self.call_from_thread(
                self.query_one("#lbl-genome-detail", Label).update,
                f"[red]Error: FASTA file not found: {fasta_path}[/red]",
            )
            self.call_from_thread(self._finish_pipeline, False)
            return

        try:
            reader = FastaReader(fasta_path)
            meta = reader.get_genome_metadata()
            total_genome_bp = meta.total_length

            # Configure Genome Progress Bar
            genome_prog = self.query_one("#genome-progress", ProgressBar)
            chr_prog = self.query_one("#chr-progress", ProgressBar)
            table = self.query_one("#chunk-table", ChunkStreamTable)

            self.call_from_thread(genome_prog.update, total=total_genome_bp, progress=0)

            chunker = SequenceChunker(chunk_size=chunk_size, overlap=overlap)
            cum_bp_processed = 0
            chunk_global_id = 0

            # Stream chromosomes
            for chrom in reader.stream_chromosomes():
                if not self.chunking_active:
                    break

                chr_name = chrom.record_id
                chr_len = chrom.metadata.length

                self.call_from_thread(
                    self.query_one("#stat-active-chr", StatCard).update_values,
                    chr_name,
                    f"Length: {chr_len:,} bp",
                )
                self.call_from_thread(
                    self.query_one("#lbl-chr-progress", Label).update,
                    f"Active Chromosome: {chr_name} ({chr_len:,} bp)",
                )
                self.call_from_thread(chr_prog.update, total=chr_len, progress=0)

                # Iterate through chunks in this chromosome
                chr_processed = 0
                for chunk in chunker.iter_chunks(chr_name, chrom.sequence):
                    # Check pause
                    while self.chunking_paused and self.chunking_active:
                        time.sleep(0.1)

                    if not self.chunking_active:
                        break

                    chunk_global_id += 1
                    cum_bp_processed = min(cum_bp_processed + chunker.step, total_genome_bp)
                    chr_processed = min(chunk.end_pos, chr_len)

                    # Calculate throughput and ETA
                    elapsed = max(time.perf_counter() - self.pipeline_start_time, 0.001)
                    throughput = cum_bp_processed / elapsed
                    chunks_per_sec = chunk_global_id / elapsed
                    remaining_bp = max(total_genome_bp - cum_bp_processed, 0)
                    eta_sec = remaining_bp / throughput if throughput > 0 else 0
                    eta_str = f"{int(eta_sec // 60):02d}:{int(eta_sec % 60):02d}"

                    pct_genome = (cum_bp_processed / total_genome_bp) * 100.0 if total_genome_bp else 0
                    pct_chr = (chr_processed / chr_len) * 100.0 if chr_len else 0

                    total_vram_kb = (chunk_global_id * chunk_size) / 1024.0

                    # UI updates
                    self.call_from_thread(genome_prog.update, progress=cum_bp_processed)
                    self.call_from_thread(chr_prog.update, progress=chr_processed)

                    self.call_from_thread(
                        self.query_one("#lbl-genome-detail", Label).update,
                        f"{cum_bp_processed:,} / {total_genome_bp:,} bp ({pct_genome:.1f}%) | ETA: {eta_str}",
                    )
                    self.call_from_thread(
                        self.query_one("#lbl-chr-detail", Label).update,
                        f"{chr_processed:,} / {chr_len:,} bp ({pct_chr:.1f}%)",
                    )

                    # Update Cards
                    self.call_from_thread(
                        self.query_one("#stat-total-bp", StatCard).update_values,
                        f"{cum_bp_processed:,} bp",
                        f"{pct_genome:.1f}% of genome",
                    )
                    self.call_from_thread(
                        self.query_one("#stat-chunks", StatCard).update_values,
                        f"{chunk_global_id:,}",
                        f"{total_vram_kb:,.1f} KB in VRAM",
                    )
                    self.call_from_thread(
                        self.query_one("#stat-throughput", StatCard).update_values,
                        f"{throughput:,.0f} bp/s",
                        f"{chunks_per_sec:.2f} chunks/s",
                    )

                    # Append to table
                    self.call_from_thread(
                        table.append_chunk,
                        chunk_id=chunk_global_id,
                        chromosome=chunk.chromosome,
                        start_pos=chunk.start_pos,
                        end_pos=chunk.end_pos,
                        length=chunk.length,
                        gc_content=chunk.gc_content,
                        ambiguous_count=chunk.ambiguous_count,
                        status="Ready",
                    )

                    # Small throttle so UI renders smoothly
                    time.sleep(0.04)

            # Mark 100% complete
            if self.chunking_active:
                self.call_from_thread(genome_prog.update, progress=total_genome_bp)
                self.call_from_thread(
                    self.query_one("#lbl-genome-detail", Label).update,
                    f"{total_genome_bp:,} / {total_genome_bp:,} bp (100.0%) | Complete ✓",
                )
                self.call_from_thread(
                    self.query_one("#stat-active-chr", StatCard).update_values,
                    "Complete",
                    f"All {meta.chromosome_count} contigs parsed",
                )
                self.call_from_thread(self._finish_pipeline, True)

        except Exception as e:
            self.call_from_thread(
                self.query_one("#lbl-genome-detail", Label).update,
                f"[red]Error: {str(e)}[/red]",
            )
            self.call_from_thread(self._finish_pipeline, False)

    def _finish_pipeline(self, success: bool) -> None:
        """Pipeline completion cleanup."""
        self.chunking_active = False
        btn_start = self.query_one("#btn-start", Button)
        btn_start.disabled = False
        btn_start.label = "↺ Re-run Chunking" if success else "▶ Retry"


def run_dashboard():
    """Launch the GeneWeaver Textual Interface."""
    app = GeneWeaverApp()
    app.run()


if __name__ == "__main__":
    run_dashboard()
