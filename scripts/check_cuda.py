"""
CUDA Device and Environment Diagnostic Utility for GeneWeaver.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cuda_engine.device_manager import DeviceManager


def main():
    print("=" * 60)
    print(" GeneWeaver CUDA Diagnostic Utility")
    print("=" * 60)

    is_cuda_ready = DeviceManager.initialize()
    print(f"[+] CUDA Runtime Initialized: {is_cuda_ready}")

    is_sim = DeviceManager.is_simulator()
    print(f"[+] Mode: {'Numba CUDA Simulator' if is_sim else 'Hardware CUDA Execution'}")

    info = DeviceManager.get_device_info(0)
    print(f"\n[+] Active Device Information:")
    print(f"    - Name               : {info.name}")
    print(f"    - Compute Capability : {info.compute_capability[0]}.{info.compute_capability[1]}")
    print(f"    - Total VRAM         : {info.total_memory_mb:,.1f} MB ({info.total_memory_bytes:,} bytes)")
    print(f"    - Simulator Active   : {info.is_simulator}")
    print("=" * 60)


if __name__ == "__main__":
    main()
