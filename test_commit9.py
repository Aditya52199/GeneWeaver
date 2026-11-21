
import unittest
from unittest.mock import patch

from geneweaver import GenomicChunk, scan_chunk

from benchmark_backends import (
    candidate_signature,
    compare_results,
    calculate_throughput,
    measure_execution,
    benchmark_chunk
)


GUIDE = b"ACGTACGTACGTACGTACGT"


def make_test_chunk():
    sequence = (
        b"A" * 10
        + GUIDE
        + b"AGG"
        + b"A" * 30
    )

    return GenomicChunk(
        chromosome="test_chromosome",
        start_position=100,
        sequence=sequence,
        valid_starts=len(sequence) - 22,
        chunk_index=0
    )


class TestCommit9(unittest.TestCase):

    def test_candidate_signature(self):
        chunk = make_test_chunk()

        candidates, stats = scan_chunk(
            chunk,
            GUIDE,
            3
        )

        self.assertGreater(len(candidates), 0)

        signature = candidate_signature(candidates[0])

        self.assertEqual(len(signature), 13)

    def test_identical_results_comparison(self):
        chunk = make_test_chunk()

        candidates, stats = scan_chunk(
            chunk,
            GUIDE,
            3
        )

        result = compare_results(
            candidates,
            candidates.copy()
        )

        self.assertTrue(result["results_match"])
        self.assertEqual(result["missing_candidates"], 0)
        self.assertEqual(result["extra_candidates"], 0)

    def test_different_results_comparison(self):
        chunk = make_test_chunk()

        candidates, stats = scan_chunk(
            chunk,
            GUIDE,
            3
        )

        self.assertGreater(len(candidates), 0)

        result = compare_results(
            candidates,
            candidates[1:]
        )

        self.assertFalse(result["results_match"])

    def test_throughput_calculation(self):
        throughput = calculate_throughput(
            total_windows=1000,
            elapsed_seconds=2
        )

        self.assertEqual(throughput, 500)

    def test_zero_time_throughput(self):
        throughput = calculate_throughput(
            total_windows=1000,
            elapsed_seconds=0
        )

        self.assertEqual(throughput, 0)

    def test_execution_measurement(self):
        measurement = measure_execution(
            lambda: 100,
            repeats=3
        )

        self.assertEqual(measurement["result"], 100)
        self.assertGreaterEqual(
            measurement["average_seconds"],
            0
        )
        self.assertGreaterEqual(
            measurement["peak_python_memory_bytes"],
            0
        )
        self.assertEqual(
            len(measurement["individual_times_seconds"]),
            3
        )

    def test_cpu_only_benchmark(self):
        chunk = make_test_chunk()

        expected_candidates, expected_stats = scan_chunk(
            chunk,
            GUIDE,
            3
        )

        fake_backend_result = {
            "backend": "CPU",
            "candidates": expected_candidates,
            "statistics": expected_stats
        }

        with patch(
            "benchmark_backends.get_backend_status",
            return_value={
                "cuda_available": False,
                "simulator_enabled": False
            }
        ):
            with patch(
                "benchmark_backends.scan_chunk_accelerated",
                return_value=fake_backend_result
            ):
                result = benchmark_chunk(
                    chunk=chunk,
                    guide=GUIDE,
                    maximum_mismatches=3,
                    repeats=1,
                    max_batch_starts=5
                )

        self.assertEqual(
            result["execution_mode"],
            "CPU_ONLY"
        )

        self.assertIsNone(result["speedup"])

        self.assertTrue(
            result["result_validation"]["results_match"]
        )

    def test_cpu_benchmark_does_not_claim_gpu_speedup(self):
        chunk = make_test_chunk()

        expected_candidates, expected_stats = scan_chunk(
            chunk,
            GUIDE,
            3
        )

        fake_backend_result = {
            "backend": "CPU",
            "candidates": expected_candidates,
            "statistics": expected_stats
        }

        with patch(
            "benchmark_backends.get_backend_status",
            return_value={
                "cuda_available": False,
                "simulator_enabled": False
            }
        ):
            with patch(
                "benchmark_backends.scan_chunk_accelerated",
                return_value=fake_backend_result
            ):
                result = benchmark_chunk(
                    chunk,
                    GUIDE,
                    3,
                    repeats=1,
                    max_batch_starts=5
                )

        self.assertIsNone(result["speedup"])
        self.assertNotEqual(
            result["execution_mode"],
            "REAL_CUDA_DEVICE"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
