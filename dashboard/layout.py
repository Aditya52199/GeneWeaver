"""
Main Layout and Tabbed Structure for GeneWeaver Textual Interface.
"""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Label, Static, TabbedContent, TabPane

from cuda_engine.device_manager import DeviceManager
from dashboard.alignment_view import AlignmentView
from dashboard.progress_view import ProgressView


class HardwareView(Widget):
    """System hardware and CUDA GPU configuration viewer."""

    def compose(self) -> ComposeResult:
        with Vertical(classes="hw-container"):
            with Vertical(classes="hw-card"):
                yield Label("GPU Acceleration & Compute Environment", classes="hw-title")

                dev_info = DeviceManager.get_device_info(0)

                with Horizontal(classes="hw-row"):
                    yield Label("Active Device Name:", classes="hw-label")
                    yield Label(f"{dev_info.name}", classes="hw-val")

                with Horizontal(classes="hw-row"):
                    yield Label("Compute Capability:", classes="hw-label")
                    yield Label(f"{dev_info.compute_capability[0]}.{dev_info.compute_capability[1]}", classes="hw-val")

                with Horizontal(classes="hw-row"):
                    yield Label("Dedicated VRAM:", classes="hw-label")
                    yield Label(f"{dev_info.total_memory_mb:,.1f} MB ({dev_info.total_memory_bytes:,} bytes)", classes="hw-val")

                with Horizontal(classes="hw-row"):
                    yield Label("CUDA Runtime Mode:", classes="hw-label")
                    mode_str = "Numba CUDA Simulator (CUDASIM)" if dev_info.is_simulator else "Hardware CUDA Native"
                    yield Label(f"{mode_str}", classes="hw-val")

            with Vertical(classes="hw-card"):
                yield Label("Genomic Engine Specifications", classes="hw-title")

                with Horizontal(classes="hw-row"):
                    yield Label("Parser Engine:", classes="hw-label")
                    yield Label("BioPython 1.88 + Streaming Reader", classes="hw-val")

                with Horizontal(classes="hw-row"):
                    yield Label("Chunk Array Layout:", classes="hw-label")
                    yield Label("Contiguous NumPy 1D (dtype=uint8 ASCII)", classes="hw-val")

                with Horizontal(classes="hw-row"):
                    yield Label("Boundary Overlap:", classes="hw-label")
                    yield Label("100 bp (PAM & Boundary Guard)", classes="hw-val")


class MainLayout(Widget):
    """Main tabbed layout of the GeneWeaver application."""

    def compose(self) -> ComposeResult:
        with TabbedContent(initial="tab-chunking"):
            with TabPane("🧬 Genomic Chunking Pipeline", id="tab-chunking"):
                yield ProgressView(id="progress-view")

            with TabPane("⚡ Sequence Alignment", id="tab-alignment"):
                yield AlignmentView(id="alignment-view")

            with TabPane("🖥 Hardware & Diagnostics", id="tab-hardware"):
                yield HardwareView(id="hardware-view")
