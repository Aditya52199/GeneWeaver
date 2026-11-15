
import unittest
from unittest.mock import patch, MagicMock

from gpu_environment import detect_cuda, decode_device_name


class TestCommit5(unittest.TestCase):

    def test_decode_bytes_device_name(self):
        result = decode_device_name(b"NVIDIA Test GPU")

        self.assertEqual(result, "NVIDIA Test GPU")

    def test_decode_string_device_name(self):
        result = decode_device_name("Test GPU")

        self.assertEqual(result, "Test GPU")

    @patch("gpu_environment.importlib.metadata.version")
    def test_numba_not_installed(self, mock_version):
        from importlib.metadata import PackageNotFoundError

        mock_version.side_effect = PackageNotFoundError

        report = detect_cuda()

        self.assertFalse(report["numba"]["installed"])
        self.assertFalse(report["cuda"]["available"])
        self.assertEqual(report["execution_mode"], "CPU")

    @patch("gpu_environment.importlib.import_module")
    @patch("gpu_environment.importlib.metadata.version")
    def test_cuda_unavailable(
        self,
        mock_version,
        mock_import
    ):
        mock_version.return_value = "test-version"

        fake_cuda = MagicMock()
        fake_cuda.is_available.return_value = False

        mock_import.return_value = fake_cuda

        report = detect_cuda()

        self.assertTrue(report["numba"]["installed"])
        self.assertFalse(report["cuda"]["available"])
        self.assertEqual(
            report["cuda"]["status"],
            "CUDA device unavailable"
        )

    @patch("gpu_environment.importlib.import_module")
    @patch("gpu_environment.importlib.metadata.version")
    def test_cuda_device_detected(
        self,
        mock_version,
        mock_import
    ):
        mock_version.return_value = "test-version"

        fake_device = MagicMock()
        fake_device.name = b"NVIDIA Test GPU"
        fake_device.compute_capability = (8, 6)
        fake_device.total_memory = 8 * 1024 * 1024 * 1024
        fake_device.MULTIPROCESSOR_COUNT = 20

        fake_cuda = MagicMock()
        fake_cuda.is_available.return_value = True
        fake_cuda.get_current_device.return_value = fake_device

        mock_import.return_value = fake_cuda

        report = detect_cuda()

        self.assertTrue(report["cuda"]["available"])
        self.assertEqual(
            report["cuda"]["gpu_name"],
            "NVIDIA Test GPU"
        )
        self.assertEqual(
            report["cuda"]["compute_capability"],
            [8, 6]
        )
        self.assertEqual(
            report["cuda"]["total_memory_mb"],
            8192.0
        )
        self.assertEqual(
            report["execution_mode"],
            "CUDA-capable"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
