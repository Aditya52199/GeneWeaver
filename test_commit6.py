
import unittest
from unittest.mock import patch

import cuda_backend

from geneweaver import (
    GenomicChunk,
    scan_chunk,
)


GUIDE = b"ACGTACGTACGTACGTACGT"


def make_test_chunk():
    sequence = (
        b"A" * 10
        + GUIDE
        + b"AGG"
        + b"A" * 20
    )

    return GenomicChunk(
        chromosome="test_chromosome",
        start_position=100,
        sequence=sequence,
        valid_starts=len(sequence) - 22,
        chunk_index=0
    )


def candidate_signature(results):
    return sorted(
        (
            item["chromosome"],
            item["start_position"],
            item["end_position"],
            item["strand"],
            item["target"],
            item["pam"],
            item["mismatch_count"],
            item["mismatch_positions"]
        )
        for item in results
    )


class TestCommit6(unittest.TestCase):

    def test_backend_status(self):
        status = cuda_backend.get_backend_status()

        self.assertIn("cuda_available", status)
        self.assertIn("execution_backend", status)

    def test_cpu_fallback(self):
        chunk = make_test_chunk()

        with patch.object(
            cuda_backend,
            "CUDA_AVAILABLE",
            False
        ):
            result = cuda_backend.scan_chunk_accelerated(
                chunk,
                GUIDE,
                3
            )

        self.assertEqual(result["backend"], "CPU")
        self.assertIn("candidates", result)
        self.assertIn("statistics", result)

    def test_cuda_matches_cpu(self):
        if not cuda_backend.CUDA_AVAILABLE:
            self.skipTest(
                "CUDA is unavailable. Run with "
                "NUMBA_ENABLE_CUDASIM=1 to test the kernel."
            )

        chunk = make_test_chunk()

        cpu_candidates, cpu_stats = scan_chunk(
            chunk,
            GUIDE,
            3
        )

        cuda_result = cuda_backend.scan_chunk_accelerated(
            chunk,
            GUIDE,
            3
        )

        self.assertEqual(cuda_result["backend"], "CUDA")

        self.assertEqual(
            candidate_signature(cpu_candidates),
            candidate_signature(cuda_result["candidates"])
        )

        self.assertEqual(
            cpu_stats["pam_sites"],
            cuda_result["statistics"]["pam_sites"]
        )

    def test_cuda_exact_match(self):
        if not cuda_backend.CUDA_AVAILABLE:
            self.skipTest(
                "CUDA is unavailable. Run with "
                "NUMBA_ENABLE_CUDASIM=1 to test the kernel."
            )

        chunk = make_test_chunk()

        result = cuda_backend.scan_chunk_accelerated(
            chunk,
            GUIDE,
            0
        )

        exact_matches = [
            candidate
            for candidate in result["candidates"]
            if candidate["mismatch_count"] == 0
        ]

        self.assertTrue(len(exact_matches) >= 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
