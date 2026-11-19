
import unittest
from unittest.mock import patch

from gpu_memory import GPUMemoryManager
import cuda_backend

from geneweaver import GenomicChunk, scan_chunk


GUIDE = b"ACGTACGTACGTACGTACGT"


def make_chunk():
    sequence = (
        b"A" * 10
        + GUIDE
        + b"AGG"
        + b"A" * 30
        + GUIDE
        + b"TGG"
    )

    return GenomicChunk(
        chromosome="test_chromosome",
        start_position=100,
        sequence=sequence,
        valid_starts=len(sequence) - 22,
        chunk_index=0
    )


class FakeCudaContext:

    def __init__(self, free_memory):
        self.free_memory = free_memory

    def get_memory_info(self):
        return self.free_memory, self.free_memory * 2


class FakeCuda:

    def __init__(self, free_memory):
        self.context = FakeCudaContext(free_memory)

    def current_context(self):
        return self.context


class TestCommit7(unittest.TestCase):

    def test_batch_splitting(self):
        manager = GPUMemoryManager(max_batch_starts=4)

        batches = list(
            manager.iter_batches(range(10), 4)
        )

        self.assertEqual(
            batches,
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9]
            ]
        )

    def test_batch_plan_respects_configured_limit(self):
        manager = GPUMemoryManager(max_batch_starts=100)

        plan = manager.plan(
            sequence_length=1000,
            total_windows=250
        )

        # Configured maximum batch size is 100.
        self.assertEqual(plan.batch_size, 100)
        self.assertEqual(plan.number_of_batches, 3)

    def test_memory_estimation(self):
        manager = GPUMemoryManager()

        estimated = manager.estimate_batch_memory(
            sequence_length=1000,
            batch_size=10
        )

        self.assertEqual(estimated, 1120)

    def test_cuda_memory_limits_batch_size(self):
        manager = GPUMemoryManager(
            max_batch_starts=1000,
            memory_fraction=0.5
        )

        fake_cuda = FakeCuda(free_memory=10000)

        plan = manager.plan(
            sequence_length=1000,
            total_windows=1000,
            cuda_module=fake_cuda
        )

        # (10000 * 0.5 - 1020) // 10 = 398
        self.assertEqual(plan.batch_size, 398)
        self.assertEqual(plan.number_of_batches, 3)

        self.assertEqual(
            plan.memory_source,
            "CUDA free-memory estimate"
        )

    def test_cpu_fallback(self):
        chunk = make_chunk()

        expected_candidates, expected_stats = scan_chunk(
            chunk,
            GUIDE,
            3
        )

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

        self.assertEqual(
            result["candidates"],
            expected_candidates
        )

        self.assertEqual(
            result["statistics"]["pam_sites"],
            expected_stats["pam_sites"]
        )

    def test_cuda_batched_results_match_cpu(self):
        if not cuda_backend.CUDA_AVAILABLE:
            self.skipTest(
                "CUDA unavailable. Run with "
                "NUMBA_ENABLE_CUDASIM=1 to test the CUDA path."
            )

        chunk = make_chunk()

        expected_candidates, expected_stats = scan_chunk(
            chunk,
            GUIDE,
            3
        )

        result = cuda_backend.scan_chunk_accelerated(
            chunk,
            GUIDE,
            3,
            max_batch_starts=5
        )

        self.assertEqual(result["backend"], "CUDA")

        self.assertEqual(
            result["candidates"],
            expected_candidates
        )

        self.assertEqual(
            result["statistics"]["pam_sites"],
            expected_stats["pam_sites"]
        )

        self.assertGreater(
            result["statistics"]["batch_count"],
            1
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
