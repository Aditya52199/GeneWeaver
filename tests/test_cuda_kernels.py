"""
Unit tests for Numba @cuda.jit kernels, Host <-> Device memory transfers, and GPU aligners.
"""

from cuda_engine.device_manager import DeviceManager
DeviceManager.initialize()

import unittest
import numpy as np

from alignment.sequence_aligner import AlignmentType, needleman_wunsch, smith_waterman
from cuda_engine.gpu_matcher import CUDABatchCRISPRScanner, CUDASyncAligner
from cuda_engine.memory_manager import MemoryManager
from cuda_engine.sequence_encoding import decode_sequence, encode_sequence


class TestCUDAEngine(unittest.TestCase):
    """Test suite for CUDA GPU alignment kernels and memory transfers."""

    @classmethod
    def setUpClass(cls):
        DeviceManager.initialize()
        cls.gpu_aligner = CUDASyncAligner(match_score=2, mismatch_penalty=-1, gap_penalty=-2)

    def test_sequence_encoding(self):
        """Verify host DNA string encoding to uint8 and decoding back."""
        seq = "ACGTN"
        encoded = encode_sequence(seq)
        np.testing.assert_array_equal(encoded, np.array([0, 1, 2, 3, 4], dtype=np.uint8))
        decoded = decode_sequence(encoded)
        self.assertEqual(decoded, seq)

    def test_memory_manager_transfers(self):
        """Verify Host (CPU RAM) to Device (GPU VRAM) and Device to Host transfers."""
        mm = MemoryManager()
        host_arr = np.array([10, 20, 30, 40], dtype=np.int32)

        # Host -> Device
        d_arr = mm.to_device(host_arr)

        # Device -> Host
        recovered = mm.to_host(d_arr)

        np.testing.assert_array_equal(host_arr, recovered)

        summary = mm.get_transfer_summary()
        self.assertGreater(summary["total_transfers"], 0)
        self.assertEqual(summary["host_to_device_bytes"], host_arr.nbytes)
        self.assertEqual(summary["device_to_host_bytes"], host_arr.nbytes)

    def test_cuda_global_alignment_identical(self):
        """Verify @cuda.jit global alignment produces exact score for identical sequences."""
        seq = "ATGCGTAC"
        gpu_res = self.gpu_aligner.align(seq, seq, mode=AlignmentType.GLOBAL)
        cpu_res = needleman_wunsch(seq, seq, match_score=2, mismatch_penalty=-1, gap_penalty=-2)

        self.assertEqual(gpu_res.score, cpu_res.score)
        self.assertEqual(gpu_res.score, len(seq) * 2)
        self.assertEqual(gpu_res.seq1_aligned, cpu_res.seq1_aligned)
        self.assertEqual(gpu_res.seq2_aligned, cpu_res.seq2_aligned)
        self.assertEqual(gpu_res.identity_percent, 100.0)

    def test_cuda_global_alignment_mismatch(self):
        """Verify @cuda.jit global alignment with substitutions."""
        seq1 = "ATGC"
        seq2 = "ATAC"  # G -> A mismatch
        gpu_res = self.gpu_aligner.align(seq1, seq2, mode=AlignmentType.GLOBAL)
        cpu_res = needleman_wunsch(seq1, seq2, match_score=2, mismatch_penalty=-1, gap_penalty=-2)

        self.assertEqual(gpu_res.score, cpu_res.score)
        self.assertEqual(gpu_res.score, 5)

    def test_cuda_local_alignment(self):
        """Verify @cuda.jit local alignment (Smith-Waterman)."""
        seq1 = "ACGT"
        seq2 = "TTTTACGTTTTT"
        gpu_res = self.gpu_aligner.align(seq1, seq2, mode=AlignmentType.LOCAL)
        cpu_res = smith_waterman(seq1, seq2, match_score=2, mismatch_penalty=-1, gap_penalty=-2)

        self.assertEqual(gpu_res.score, cpu_res.score)
        self.assertEqual(gpu_res.seq1_aligned, "ACGT")
        self.assertEqual(gpu_res.seq2_aligned, "ACGT")

    def test_cuda_batch_crispr_scanner(self):
        """Verify batch CRISPR off-target scanner detects candidate hits with PAMs."""
        scanner = CUDABatchCRISPRScanner(max_mismatches=2)

        guide = "GTCACCTGTAGTGACGTACACGG"  # 20bp spacer + CGG PAM
        # Target chunk with an exact match and a 1-mismatch match
        target = (
            "AAAAAAAAAA"
            + guide                                # Exact match at pos 10
            + "TTTTTTTTTT"
            + "GTCACCTGTAGTGACGTACACGG"[:10] + "A" + "GTCACCTGTAGTGACGTACACGG"[11:]  # 1 mismatch
            + "CCCCCCCCCC"
        )

        hits = scanner.scan_chunk(target, guide)
        self.assertGreaterEqual(len(hits), 1)

        # Check that PAM ends in GG
        for hit in hits:
            self.assertTrue(hit.pam.endswith("GG"))
            self.assertLessEqual(hit.mismatches, 2)


if __name__ == "__main__":
    unittest.main()
