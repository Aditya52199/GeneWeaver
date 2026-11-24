"""
Alignment and Off-Target Inspection View for GeneWeaver Textual Interface.
"""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Button, Input, Label, Select, Static

from alignment.sequence_aligner import AlignmentType, SequenceAligner


class AlignmentView(Widget):
    """View allowing live testing and visualization of DNA sequence alignment."""

    def compose(self) -> ComposeResult:
        with Vertical(classes="align-container"):
            with Horizontal(classes="align-input-row"):
                yield Input(
                    value="GTCACCTGTAGTGACGTACACGG",
                    placeholder="Guide RNA (20bp spacer + 3bp PAM)...",
                    classes="control-input",
                    id="align-guide",
                )
                yield Input(
                    value="AAAAAAAAAAGTCACCTGTAGTGACGTACACGGTTTTTTTTTT",
                    placeholder="Target DNA sequence...",
                    classes="control-input",
                    id="align-target",
                )
                yield Select(
                    [
                        ("Global (Needleman-Wunsch)", AlignmentType.GLOBAL),
                        ("Local (Smith-Waterman)", AlignmentType.LOCAL),
                        ("Semi-Global (CRISPR)", AlignmentType.SEMI_GLOBAL),
                    ],
                    value=AlignmentType.SEMI_GLOBAL,
                    id="align-mode-select",
                    allow_blank=False,
                )
                yield Button("⚡ Align Sequences", classes="btn-start", id="btn-run-align")

            with Container(classes="align-display"):
                yield Static(
                    "Ready to align. Enter sequences above and press 'Align Sequences'.",
                    id="align-result-display",
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle run alignment button."""
        if event.button.id == "btn-run-align":
            self.execute_alignment()

    def execute_alignment(self) -> None:
        """Run sequence alignment and display formatted output."""
        guide_input = self.query_one("#align-guide", Input).value.strip()
        target_input = self.query_one("#align-target", Input).value.strip()
        mode_select = self.query_one("#align-mode-select", Select).value

        display = self.query_one("#align-result-display", Static)

        if not guide_input or not target_input:
            display.update("[red]Error: Both Guide RNA and Target DNA sequences must be provided.[/red]")
            return

        aligner = SequenceAligner(match_score=2, mismatch_penalty=-1, gap_penalty=-2)
        try:
            result = aligner.align(guide_input, target_input, mode=mode_select)
            viz = result.formatted_alignment(line_length=60)

            output_text = (
                f"[bold cyan]Alignment Algorithm:[/bold cyan] {result.alignment_type.upper()}\n"
                f"[bold cyan]Score:[/bold cyan] {result.score}  |  "
                f"[bold cyan]Identity:[/bold cyan] {result.identity_percent:.1f}%  |  "
                f"[bold cyan]CIGAR:[/bold cyan] {result.cigar}\n"
                f"[bold cyan]Execution Time:[/bold cyan] {result.execution_time_ms:.3f} ms  |  "
                f"[bold cyan]Speed:[/bold cyan] {result.mcups:.4f} MCUPS  |  "
                f"[bold cyan]Cells:[/bold cyan] {result.cells_computed:,}\n\n"
                f"[bold green]Alignment Visualization:[/bold green]\n"
                f"{viz}"
            )
            display.update(output_text)
        except Exception as e:
            display.update(f"[red]Alignment failed: {str(e)}[/red]")
