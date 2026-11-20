"""
GeneWeaver CUDA Engine Package.

Provides Numba @cuda.jit sequence alignment kernels, Host <-> Device memory management,
and GPU-accelerated CRISPR off-target scanners.
"""

from cuda_engine.device_manager import DeviceInfo, DeviceManager
from cuda_engine.gpu_matcher import CUDABatchCRISPRScanner, CUDASyncAligner, OffTargetHit
from cuda_engine.memory_manager import MemoryManager, TransferStats
from cuda_engine.sequence_encoding import decode_sequence, encode_sequence

__all__ = [
    "CUDABatchCRISPRScanner",
    "CUDASyncAligner",
    "DeviceInfo",
    "DeviceManager",
    "MemoryManager",
    "OffTargetHit",
    "TransferStats",
    "decode_sequence",
    "encode_sequence",
]
