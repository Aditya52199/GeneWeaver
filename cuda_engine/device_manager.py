"""
CUDA Device Management for GeneWeaver.

Handles GPU device query, capability inspection, VRAM accounting,
and seamless fallback between Hardware CUDA and Numba CUDASIM.
"""

import ctypes
from dataclasses import dataclass
import os
import sys
from typing import Dict, List, Optional, Tuple


def _ensure_cuda_environment():
    """Configure environment before importing numba.cuda."""
    if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1":
        return

    # Check whether NVVM (CUDA compiler backend) is available on the OS
    nvvm_available = False
    try:
        # Check standard Windows / Linux library load
        if sys.platform == "win32":
            ctypes.CDLL("nvvm.dll")
        else:
            ctypes.CDLL("libnvvm.so")
        nvvm_available = True
    except Exception:
        nvvm_available = False

    if not nvvm_available:
        # Enable Numba CUDA Simulator for full @cuda.jit execution fidelity
        os.environ["NUMBA_ENABLE_CUDASIM"] = "1"


# Run setup before importing numba
_ensure_cuda_environment()
from numba import cuda


@dataclass
class DeviceInfo:
    """Hardware and runtime specifications for an NVIDIA GPU."""
    device_id: int
    name: str
    compute_capability: Tuple[int, int]
    total_memory_bytes: int
    total_memory_mb: float
    is_simulator: bool

    def __str__(self) -> str:
        sim_tag = " (CUDASIM)" if self.is_simulator else ""
        return (
            f"Device {self.device_id}: {self.name}{sim_tag} | "
            f"Compute: {self.compute_capability[0]}.{self.compute_capability[1]} | "
            f"VRAM: {self.total_memory_mb:.1f} MB"
        )


class DeviceManager:
    """Manages GPU contexts, stream creation, and hardware/simulator detection."""

    _initialized = False
    _is_simulator = False

    @classmethod
    def initialize(cls) -> bool:
        """Initialize CUDA runtime or configure Numba CUDA Simulator."""
        _ensure_cuda_environment()
        cls._is_simulator = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
        cls._initialized = True
        return True

    @classmethod
    def is_simulator(cls) -> bool:
        """Check if running under Numba CUDA Simulator."""
        cls.initialize()
        return cls._is_simulator

    @classmethod
    def get_device_info(cls, device_id: int = 0) -> DeviceInfo:
        """Get device information and capabilities."""
        cls.initialize()

        if cls._is_simulator:
            return DeviceInfo(
                device_id=0,
                name="NVIDIA GeForce RTX 4050 Laptop GPU (CUDASIM Mode)",
                compute_capability=(8, 9),
                total_memory_bytes=6 * 1024 * 1024 * 1024,
                total_memory_mb=6144.0,
                is_simulator=True,
            )

        device = cuda.select_device(device_id)
        name = getattr(device, "name", "NVIDIA GPU")
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        cc = getattr(device, "compute_capability", (8, 0))
        mem_info = cuda.current_context().get_memory_info()
        total_mem = mem_info.total
        total_mb = total_mem / (1024 * 1024)

        return DeviceInfo(
            device_id=device_id,
            name=name,
            compute_capability=cc,
            total_memory_bytes=total_mem,
            total_memory_mb=total_mb,
            is_simulator=False,
        )

    @classmethod
    def create_stream(cls):
        """Create a CUDA stream for asynchronous memory copy and execution."""
        cls.initialize()
        return cuda.stream()

    @classmethod
    def synchronize(cls) -> None:
        """Block CPU until all preceding commands in all CUDA streams have finished."""
        cls.initialize()
        cuda.synchronize()
