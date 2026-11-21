"""
Host-to-Device and Device-to-Host Memory Manager for GeneWeaver.

Handles complex memory transfers between Host (CPU RAM) and Device (GPU VRAM),
including pinned memory allocation, asynchronous stream copies, and bandwidth profiling.
"""

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from cuda_engine.device_manager import DeviceManager

DeviceManager.initialize()
from numba import cuda


@dataclass
class TransferStats:
    """Metrics recorded for Host <-> Device memory transfers."""
    direction: str  # "HtoD" (Host to Device) or "DtoH" (Device to Host)
    bytes_transferred: int
    duration_ms: float
    bandwidth_gb_per_sec: float

    def __str__(self) -> str:
        return (
            f"[{self.direction}] {self.bytes_transferred:,} bytes in "
            f"{self.duration_ms:.3f} ms ({self.bandwidth_gb_per_sec:.2f} GB/s)"
        )


class MemoryManager:
    """
    Manages VRAM allocations, host pinned memory, and explicit memory transfers
    between Host (CPU RAM) and Device (GPU VRAM).
    """

    def __init__(self, use_pinned_memory: bool = True):
        """
        Initialize MemoryManager.

        Args:
            use_pinned_memory: Whether to use page-locked host memory for faster DMA.
        """
        DeviceManager.initialize()
        self.use_pinned = use_pinned_memory
        self.transfer_log: List[TransferStats] = []
        self._total_device_allocated_bytes = 0

    def allocate_pinned_host(self, shape: Union[int, Tuple[int, ...]], dtype: np.dtype) -> np.ndarray:
        """
        Allocate page-locked (pinned) Host CPU RAM.
        Allows the GPU DMA engine to transfer data directly without intermediate paging.
        """
        try:
            return cuda.pinned_array(shape, dtype=dtype)
        except Exception:
            # Fallback to standard NumPy contiguous array if pinned fails
            return np.zeros(shape, dtype=dtype)

    def allocate_device(
        self,
        shape: Union[int, Tuple[int, ...]],
        dtype: np.dtype,
        stream=None,
    ) -> Any:
        """
        Allocate uninitialized memory directly on Device VRAM.

        Args:
            shape: Array dimensions.
            dtype: Data type (e.g. np.int32, np.uint8).
            stream: Optional CUDA stream.

        Returns:
            DeviceNDArray on GPU VRAM.
        """
        arr = cuda.device_array(shape, dtype=dtype, stream=stream)
        num_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        self._total_device_allocated_bytes += num_bytes
        return arr

    def to_device(
        self,
        host_arr: np.ndarray,
        stream=None,
        record_stats: bool = True,
    ) -> Any:
        """
        Transfer memory from Host (CPU RAM) to Device (GPU VRAM).

        Args:
            host_arr: Contiguous NumPy array on Host.
            stream: Optional CUDA stream for asynchronous transfer.
            record_stats: Whether to log transfer duration and bandwidth.

        Returns:
            Device array on GPU VRAM.
        """
        if not host_arr.flags["C_CONTIGUOUS"]:
            host_arr = np.ascontiguousarray(host_arr)

        start_time = time.perf_counter()
        device_arr = cuda.to_device(host_arr, stream=stream)

        if stream is not None:
            stream.synchronize()
        else:
            DeviceManager.synchronize()

        elapsed_seconds = time.perf_counter() - start_time
        num_bytes = host_arr.nbytes
        self._total_device_allocated_bytes += num_bytes

        if record_stats:
            duration_ms = elapsed_seconds * 1000.0
            bandwidth_gb = (num_bytes / 1e9) / elapsed_seconds if elapsed_seconds > 0 else 0.0
            stat = TransferStats(
                direction="Host->Device",
                bytes_transferred=num_bytes,
                duration_ms=round(duration_ms, 3),
                bandwidth_gb_per_sec=round(bandwidth_gb, 3),
            )
            self.transfer_log.append(stat)

        return device_arr

    def to_host(
        self,
        device_arr: Any,
        host_dest: Optional[np.ndarray] = None,
        stream=None,
        record_stats: bool = True,
    ) -> np.ndarray:
        """
        Transfer memory from Device (GPU VRAM) back to Host (CPU RAM).

        Args:
            device_arr: DeviceNDArray residing on GPU.
            host_dest: Optional destination NumPy array on Host. If None, allocates one.
            stream: Optional CUDA stream.
            record_stats: Whether to log transfer duration and bandwidth.

        Returns:
            NumPy array in Host CPU RAM.
        """
        start_time = time.perf_counter()

        if host_dest is None:
            result = device_arr.copy_to_host(stream=stream)
        else:
            device_arr.copy_to_host(host_dest, stream=stream)
            result = host_dest

        if stream is not None:
            stream.synchronize()
        else:
            DeviceManager.synchronize()

        elapsed_seconds = time.perf_counter() - start_time
        num_bytes = result.nbytes

        if record_stats:
            duration_ms = elapsed_seconds * 1000.0
            bandwidth_gb = (num_bytes / 1e9) / elapsed_seconds if elapsed_seconds > 0 else 0.0
            stat = TransferStats(
                direction="Device->Host",
                bytes_transferred=num_bytes,
                duration_ms=round(duration_ms, 3),
                bandwidth_gb_per_sec=round(bandwidth_gb, 3),
            )
            self.transfer_log.append(stat)

        return result

    def get_transfer_summary(self) -> Dict[str, Any]:
        """Produce a summary of all memory transfers executed."""
        htod = [s for s in self.transfer_log if s.direction == "Host->Device"]
        dtoh = [s for s in self.transfer_log if s.direction == "Device->Host"]

        htod_bytes = sum(s.bytes_transferred for s in htod)
        dtoh_bytes = sum(s.bytes_transferred for s in dtoh)
        htod_time = sum(s.duration_ms for s in htod)
        dtoh_time = sum(s.duration_ms for s in dtoh)

        return {
            "total_transfers": len(self.transfer_log),
            "host_to_device_bytes": htod_bytes,
            "host_to_device_time_ms": round(htod_time, 3),
            "device_to_host_bytes": dtoh_bytes,
            "device_to_host_time_ms": round(dtoh_time, 3),
            "total_transferred_mb": round((htod_bytes + dtoh_bytes) / (1024 * 1024), 2),
            "active_vram_bytes": self._total_device_allocated_bytes,
        }

    def reset_stats(self) -> None:
        """Clear transfer logs and metrics."""
        self.transfer_log.clear()
        self._total_device_allocated_bytes = 0
