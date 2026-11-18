"""
Unit tests for Pure Python DNA Sequence Aligner and CPU Baseline Benchmark.
"""

import unittest
from alignment.sequence_aligner import (
    AlignmentResult,
    AlignmentType,
    SequenceAligner,
    needleman_wunsch,
    smith_waterman,
)
from benchmarks.cpu_benchmark import (
    CPUBenchmark,
    generate_dna_sequence,
    mutate_dna_sequence,
)


class TestSequenceAligner(unittest.TestCase):
    """Test suite for Needleman-Wunsch, Smith-Waterman, and Semi-Global alignments."""

    def setUp(self):
        self.aligner = SequenceAligner(match_score=2, mismatch_penalty=-1, gap_penalty=-2)

    def test_global_identical_sequences(self):
        """Identical sequences should align with 100% identity and maximum match score."""
        seq = "ATGCGTAC"
        res = needleman_wunsch(seq, seq)

        self.assertEqual(res.seq1_aligned, seq)
        self.assertEqual(res.seq2_aligned, seq)
        self.assertEqual(res.score, len(seq) * 2)
        self.assertEqual(res.identity_percent, 100.0)
        self.assertEqual(res.matches, len(seq))
        self.assertEqual(res.mismatches, 0)
        self.assertEqual(res.gaps, 0)
        self.assertEqual(res.cigar, f"{len(seq)}M")

    def test_global_single_mismatch(self):
        """Single mismatch should deduct points accordingly."""
        seq1 = "ATGC"
        seq2 = "ATAC"  # G -> A mismatch at pos 2
        res = needleman_wunsch(seq1, seq2, match_score=2, mismatch_penalty=-1, gap_penalty=-2)

        # 3 matches (+6), 1 mismatch (-1) = 5
        self.assertEqual(res.score, 5)
        self.assertEqual(res.matches, 3)
        self.assertEqual(res.mismatches, 1)
        self.assertEqual(res.gaps, 0)
        self.assertEqual(res.identity_percent, 75.0)

    def test_global_with_gap(self):
        """Insertion or deletion should be handled properly."""
        seq1 = "AGTC"
        seq2 = "AGC"  # Deletion of T
        res = needleman_wunsch(seq1, seq2, match_score=2, mismatch_penalty=-1, gap_penalty=-2)

        # 3 matches (+6), 1 gap (-2) = 4
        self.assertEqual(res.score, 4)
        self.assertEqual(res.matches, 3)
        self.assertEqual(res.gaps, 1)

    def test_local_smith_waterman(self):
        """Smith-Waterman should identify local high-scoring regions."""
        # Seq2 has the exact motif "ACGT" surrounded by non-matching letters
        seq1 = "ACGT"
        seq2 = "TTTTACGTTTTT"
        res = smith_waterman(seq1, seq2, match_score=2, mismatch_penalty=-1, gap_penalty=-2)

        self.assertEqual(res.seq1_aligned, "ACGT")
        self.assertEqual(res.seq2_aligned, "ACGT")
        self.assertEqual(res.score, 8)  # 4 * 2
        self.assertEqual(res.identity_percent, 100.0)
        self.assertEqual(res.start_pos, (0, 4))
        self.assertEqual(res.end_pos, (4, 8))

    def test_semi_global_alignment(self):
        """Semi-global alignment should allow free end gaps in target."""
        seq1 = "ACGT"
        seq2 = "NNNNACGTTTT"
        res = self.aligner.align(seq1, seq2, mode=AlignmentType.SEMI_GLOBAL)

        self.assertEqual(res.seq1_aligned, "ACGT")
        self.assertEqual(res.seq2_aligned, "ACGT")
        self.assertEqual(res.score, 8)

    def test_empty_sequence(self):
        """Edge case with empty sequence."""
        res = needleman_wunsch("", "")
        self.assertEqual(res.score, 0)
        self.assertEqual(res.seq1_aligned, "")
        self.assertEqual(res.seq2_aligned, "")

    def test_formatted_alignment_output(self):
        """Visualization helper generates 3-line format."""
        seq1 = "ACGT"
        seq2 = "ACTT"
        res = needleman_wunsch(seq1, seq2)
        formatted = res.formatted_alignment()
        self.assertIn("Seq1: ACGT", formatted)
        self.assertIn("      ||.|", formatted)
        self.assertIn("Seq2: ACTT", formatted)

    def test_metric_calculations(self):
        """Verify performance metrics such as cells_computed and MCUPS."""
        seq1 = "A" * 10
        seq2 = "A" * 10
        res = needleman_wunsch(seq1, seq2)

        expected_cells = 11 * 11
        self.assertEqual(res.cells_computed, expected_cells)
        self.assertGreaterEqual(res.execution_time_seconds, 0.0)
        self.assertGreater(res.mcups, 0.0)


class TestBenchmarkSuite(unittest.TestCase):
    """Test suite for benchmark data generator and runner."""

    def test_generate_dna_sequence(self):
        """DNA generator generates valid bases with correct length."""
        length = 50
        seq = generate_dna_sequence(length, gc_content=0.5, seed=42)
        self.assertEqual(len(seq), length)
        self.assertTrue(all(c in "ACGT" for c in seq))

    def test_mutate_dna_sequence(self):
        """DNA mutator modifies sequence."""
        seq = "A" * 100
        mutated = mutate_dna_sequence(seq, mutation_rate=0.2, seed=42)
        self.assertNotEqual(seq, mutated)
        self.assertTrue(any(c != "A" for c in mutated))

    def test_cpu_benchmark_runner(self):
        """Run a mini benchmark suite to verify metrics collection."""
        runner = CPUBenchmark(warmup_runs=1)
        summary = runner.run_scaling_suite(
            sequence_lengths=[20, 40],
            mode=AlignmentType.GLOBAL,
            trials=2,
        )

        self.assertEqual(len(summary.results), 2)
        self.assertGreater(summary.overall_avg_mcups, 0.0)
        self.assertGreater(summary.results[0].mean_time_ms, 0.0)
        self.assertGreater(summary.results[0].throughput_bp_per_sec, 0.0)


if __name__ == "__main__":
    unittest.main()

