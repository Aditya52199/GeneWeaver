"""
GPU vs CPU Benchmark for GeneWeaver Sequence Alignment.

Benchmarks Numba @cuda.jit kernels, Host <-> Device memory transfer overheads,
and alignment throughput.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alignment.sequence_aligner import AlignmentType, needleman_wunsch
from benchmarks.cpu_benchmark import generate_dna_sequence, mutate_dna_sequence
from cuda_engine import CUDASyncAligner, DeviceManager, MemoryManager


def run_gpu_benchmark(lengths=[50, 100, 200, 400], trials=3):
    print("=" * 80)
    print(" GeneWeaver: GPU vs CPU Sequence Alignment Benchmark")
    print("=" * 80)

    DeviceManager.initialize()
    dev_info = DeviceManager.get_device_info(0)
    print(f"Device: {dev_info.name} (Compute {dev_info.compute_capability[0]}.{dev_info.compute_capability[1]})")
    print(f"Mode: {'CUDASIM' if dev_info.is_simulator else 'Hardware CUDA'}\n")

    aligner_gpu = CUDASyncAligner()

    print(f" {'Length (bp)':<12} | {'Cells':<12} | {'CPU Time (ms)':<15} | {'GPU Time (ms)':<15} | {'H<->D Transfers':<16}")
    print("-" * 80)

    for length in lengths:
        seq1 = generate_dna_sequence(length, seed=42 + length)
        seq2 = mutate_dna_sequence(seq1, mutation_rate=0.05, seed=100 + length)

        # CPU Baseline
        cpu_start = time.perf_counter()
        for _ in range(trials):
            cpu_res = needleman_wunsch(seq1, seq2)
        cpu_time_ms = (time.perf_counter() - cpu_start) / trials * 1000.0

        # GPU Run
        aligner_gpu.memory_manager.reset_stats()
        gpu_start = time.perf_counter()
        for _ in range(trials):
            gpu_res = aligner_gpu.align(seq1, seq2, mode=AlignmentType.GLOBAL)
        gpu_time_ms = (time.perf_counter() - gpu_start) / trials * 1000.0

        cells = (length + 1) * (length + 1)
        transfer_summary = aligner_gpu.memory_manager.get_transfer_summary()
        hd_time = transfer_summary["host_to_device_time_ms"] + transfer_summary["device_to_host_time_ms"]

        print(
            f" {length:<12} | {cells:<12,d} | {cpu_time_ms:<15.3f} | {gpu_time_ms:<15.3f} | "
            f"{hd_time:.2f} ms total"
        )

        assert gpu_res.score == cpu_res.score, f"Score mismatch: GPU={gpu_res.score}, CPU={cpu_res.score}"

    print("=" * 80)
    print("[+] All GPU alignments verified exact match with CPU baseline scores.")


def main():
    parser = argparse.ArgumentParser(description="GeneWeaver GPU Benchmark")
    parser.add_argument("--lengths", nargs="+", type=int, default=[50, 100, 200], help="Sequence lengths")
    parser.add_argument("--trials", type=int, default=2, help="Trials per length")
    args = parser.parse_args()

    run_gpu_benchmark(lengths=args.lengths, trials=args.trials)


if __name__ == "__main__":
    main()
