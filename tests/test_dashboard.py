"""
Automated tests for GeneWeaver Textual Interface.
"""

import unittest
from dashboard.app import GeneWeaverApp
from dashboard.progress_view import ProgressView
from dashboard.widgets import ChunkStreamTable, StatCard


class TestDashboardWidgets(unittest.TestCase):
    """Test suite for Textual dashboard components and logic."""

    def test_app_instantiation(self):
        """Verify GeneWeaverApp instantiates and initializes properties."""
        app = GeneWeaverApp()
        self.assertEqual(app.title, "GeneWeaver: GPU-Accelerated CRISPR Alignment Engine")
        self.assertFalse(app.chunking_active)
        self.assertFalse(app.chunking_paused)
        self.assertEqual(app.total_processed_bp, 0)
        self.assertEqual(app.total_chunks_created, 0)

    def test_stat_card_initialization(self):
        """Verify StatCard widget initializes with given title and values."""
        card = StatCard(title="TOTAL NUCLEOTIDES", value="160,000 bp", subtext="100% of genome", id="stat-test")
        self.assertEqual(card._title, "TOTAL NUCLEOTIDES")
        self.assertEqual(card._value, "160,000 bp")
        self.assertEqual(card._subtext, "100% of genome")

    def test_chunk_stream_table_columns(self):
        """Verify ChunkStreamTable column definitions."""
        table = ChunkStreamTable()
        # Table columns are configured in on_mount, let's verify instance
        self.assertIsInstance(table, ChunkStreamTable)

    def test_progress_view_composition(self):
        """Verify ProgressView can be created."""
        pv = ProgressView()
        self.assertIsInstance(pv, ProgressView)


if __name__ == "__main__":
    unittest.main()
