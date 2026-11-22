
import unittest
from unittest.mock import patch

import gpu_batch_processor as batch


class FakeChunk:
    def __init__(self, chunk_index):
        self.chunk_index = chunk_index


class TestCommit10(unittest.TestCase):

    def setUp(self):
        self.chunks = [
            FakeChunk(0),
            FakeChunk(1),
            FakeChunk(2)
        ]

    @patch("gpu_batch_processor.get_backend_status")
    def test_cpu_backend_selection(self, mock_status):
        mock_status.return_value = {
            "cuda_available": False,
            "simulator_enabled": False
        }

        function, name = batch.select_backend("auto")

        self.assertEqual(name, "CPU")
        self.assertIs(function, batch.scan_chunk)

    @patch("gpu_batch_processor.get_backend_status")
    def test_simulator_backend_selection(self, mock_status):
        mock_status.return_value = {
            "cuda_available": False,
            "simulator_enabled": True
        }

        function, name = batch.select_backend("auto")

        self.assertEqual(name, "CUDA_SIMULATOR")
        self.assertIs(function, batch.scan_chunk_accelerated)

    @patch("gpu_batch_processor.get_backend_status")
    def test_hardware_backend_selection(self, mock_status):
        mock_status.return_value = {
            "cuda_available": True,
            "simulator_enabled": False
        }

        function, name = batch.select_backend("auto")

        self.assertEqual(name, "CUDA")
        self.assertIs(function, batch.scan_chunk_accelerated)

    @patch("gpu_batch_processor.get_backend_status")
    def test_cpu_batch_processing(self, mock_status):
        mock_status.return_value = {
            "cuda_available": False,
            "simulator_enabled": False
        }

        candidates = [
            {"position": 100},
            {"position": 200}
        ]

        def fake_cpu_scan(chunk, guide, mismatches):
            return candidates[:1], {"windows_scanned": 10}

        with patch(
            "gpu_batch_processor.scan_chunk",
            side_effect=fake_cpu_scan
        ):
            result = batch.process_chunks(
                chunks=self.chunks,
                guide_bytes=b"A" * 20,
                maximum_mismatches=3,
                max_batch_starts=5,
                backend="cpu"
            )

        self.assertEqual(result["backend"], "CPU")
        self.assertEqual(result["chunks_processed"], 3)
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual(
            result["statistics"]["windows_scanned"],
            30
        )

    @patch("gpu_batch_processor.get_backend_status")
    def test_all_candidates_are_collected(self, mock_status):
        mock_status.return_value = {
            "cuda_available": False,
            "simulator_enabled": False
        }

        def fake_cpu_scan(chunk, guide, mismatches):
            return [
                {"chunk": chunk.chunk_index}
            ], {}

        with patch(
            "gpu_batch_processor.scan_chunk",
            side_effect=fake_cpu_scan
        ):
            result = batch.process_chunks(
                self.chunks,
                b"A" * 20,
                backend="cpu"
            )

        self.assertEqual(
            [item["chunk"] for item in result["candidates"]],
            [0, 1, 2]
        )

    @patch("gpu_batch_processor.get_backend_status")
    def test_empty_chunks(self, mock_status):
        mock_status.return_value = {
            "cuda_available": False,
            "simulator_enabled": False
        }

        result = batch.process_chunks(
            [],
            b"A" * 20,
            backend="cpu"
        )

        self.assertEqual(result["chunks_processed"], 0)
        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(result["candidates"], [])

    def test_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            batch.process_chunks(
                self.chunks,
                b"A" * 20,
                max_batch_starts=0,
                backend="cpu"
            )

    def test_invalid_mismatch_limit(self):
        with self.assertRaises(ValueError):
            batch.process_chunks(
                self.chunks,
                b"A" * 20,
                maximum_mismatches=-1,
                backend="cpu"
            )

    def test_valid_guide(self):
        guide = "ACGTACGTACGTACGTACGT"

        result = batch.validate_guide(guide)

        self.assertEqual(result, guide.encode("ascii"))

    def test_invalid_guide_length(self):
        with self.assertRaises(ValueError):
            batch.validate_guide("ACGT")

    def test_invalid_guide_base(self):
        with self.assertRaises(ValueError):
            batch.validate_guide("ACGTACGTACGTACGTACGX")


if __name__ == "__main__":
    unittest.main()
