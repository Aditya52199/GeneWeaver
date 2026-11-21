
import argparse
import json
import os
import time
import tracemalloc

from datetime import datetime, timezone
from pathlib import Path

from geneweaver import GenomicParser, scan_chunk
from cuda_backend import (
    scan_chunk_accelerated,
    get_backend_status
)


def candidate_signature(candidate):
    """Return fields used to compare alignment results."""

    fields = (
        "chromosome",
        "start_position",
        "end_position",
        "strand",
        "guide",
        "target",
        "pam",
        "mismatch_count",
        "mismatch_positions",
        "seed_mismatches",
        "gc_content",
        "candidate_score",
        "priority"
    )

    return tuple(candidate.get(field) for field in fields)


def compare_results(cpu_candidates, backend_candidates):
    """Check whether two backends produce identical candidate results."""

    cpu_results = sorted(
        candidate_signature(item)
        for item in cpu_candidates
    )

    backend_results = sorted(
        candidate_signature(item)
        for item in backend_candidates
    )

    return {
        "results_match": cpu_results == backend_results,
        "cpu_candidates": len(cpu_results),
        "backend_candidates": len(backend_results),
        "missing_candidates": len(
            set(cpu_results) - set(backend_results)
        ),
        "extra_candidates": len(
            set(backend_results) - set(cpu_results)
        )
    }


def measure_execution(function, repeats):
    """Measure average execution time and peak Python-tracked memory."""

    execution_times = []
    peak_memory_values = []
    final_result = None

    for _ in range(repeats):
        tracemalloc.start()

        start_time = time.perf_counter()
        final_result = function()
        elapsed = time.perf_counter() - start_time

        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        execution_times.append(elapsed)
        peak_memory_values.append(peak_memory)

    return {
        "result": final_result,
        "average_seconds": sum(execution_times) / repeats,
        "minimum_seconds": min(execution_times),
        "maximum_seconds": max(execution_times),
        "peak_python_memory_bytes": max(peak_memory_values),
        "individual_times_seconds": execution_times
    }


def calculate_throughput(total_windows, elapsed_seconds):
    if elapsed_seconds <= 0:
        return 0.0

    return total_windows / elapsed_seconds


def benchmark_chunk(
    chunk,
    guide,
    maximum_mismatches,
    repeats,
    max_batch_starts
):
    """Benchmark CPU and, when available, the selected CUDA backend."""

    cpu_measurement = measure_execution(
        lambda: scan_chunk(
            chunk,
            guide,
            maximum_mismatches
        ),
        repeats
    )

    cpu_candidates, cpu_statistics = cpu_measurement["result"]

    cpu_windows = min(
        chunk.valid_starts,
        max(0, len(chunk.sequence) - 22)
    )

    cpu_throughput = calculate_throughput(
        cpu_windows,
        cpu_measurement["average_seconds"]
    )

    backend_status = get_backend_status()
    cuda_available = backend_status["cuda_available"]

    simulator_enabled = (
        os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
    )

    if cuda_available:
        backend_measurement = measure_execution(
            lambda: scan_chunk_accelerated(
                chunk,
                guide,
                maximum_mismatches,
                max_batch_starts=max_batch_starts
            ),
            repeats
        )

        backend_result = backend_measurement["result"]

        backend_candidates = backend_result["candidates"]

        backend_throughput = calculate_throughput(
            cpu_windows,
            backend_measurement["average_seconds"]
        )

        comparison = compare_results(
            cpu_candidates,
            backend_candidates
        )

        if simulator_enabled:
            execution_mode = "CUDA_SIMULATOR"
            speedup = None
            speedup_note = (
                "Simulator timing is not real GPU performance. "
                "No hardware speedup is reported."
            )
        else:
            execution_mode = "REAL_CUDA_DEVICE"

            if backend_measurement["average_seconds"] > 0:
                speedup = (
                    cpu_measurement["average_seconds"]
                    / backend_measurement["average_seconds"]
                )
            else:
                speedup = None

            speedup_note = (
                "Measured on an accessible CUDA device. "
                "Results should be interpreted with the "
                "benchmark configuration."
            )

        backend_details = {
            "backend": backend_result["backend"],
            "average_seconds": backend_measurement["average_seconds"],
            "minimum_seconds": backend_measurement["minimum_seconds"],
            "maximum_seconds": backend_measurement["maximum_seconds"],
            "throughput_windows_per_second": backend_throughput,
            "peak_python_memory_bytes": (
                backend_measurement["peak_python_memory_bytes"]
            ),
            "individual_times_seconds": (
                backend_measurement["individual_times_seconds"]
            ),
            "candidate_count": len(backend_candidates)
        }

    else:
        execution_mode = "CPU_ONLY"

        backend_result = scan_chunk_accelerated(
            chunk,
            guide,
            maximum_mismatches,
            max_batch_starts=max_batch_starts
        )

        comparison = compare_results(
            cpu_candidates,
            backend_result["candidates"]
        )

        speedup = None
        speedup_note = (
            "No accessible CUDA device. GPU performance "
            "comparison was not performed."
        )

        backend_details = {
            "backend": "CPU_FALLBACK",
            "average_seconds": None,
            "throughput_windows_per_second": None,
            "peak_python_memory_bytes": None,
            "candidate_count": len(backend_result["candidates"])
        }

    return {
        "chunk_index": chunk.chunk_index,
        "chromosome": chunk.chromosome,
        "sequence_length": len(chunk.sequence),
        "windows_processed": cpu_windows,
        "cpu": {
            "average_seconds": cpu_measurement["average_seconds"],
            "minimum_seconds": cpu_measurement["minimum_seconds"],
            "maximum_seconds": cpu_measurement["maximum_seconds"],
            "throughput_windows_per_second": cpu_throughput,
            "peak_python_memory_bytes": (
                cpu_measurement["peak_python_memory_bytes"]
            ),
            "individual_times_seconds": (
                cpu_measurement["individual_times_seconds"]
            ),
            "candidate_count": len(cpu_candidates),
            "pam_sites": cpu_statistics["pam_sites"]
        },
        "selected_backend": backend_details,
        "execution_mode": execution_mode,
        "speedup": speedup,
        "speedup_note": speedup_note,
        "result_validation": comparison
    }


def run_benchmark(
    fasta_path,
    guide_text,
    maximum_mismatches,
    chunk_size,
    repeats,
    max_batch_starts,
    max_chunks
):
    guide_text = guide_text.upper()

    if len(guide_text) != 20:
        raise ValueError("Guide must contain exactly 20 bases.")

    if any(base not in "ACGT" for base in guide_text):
        raise ValueError("Guide must contain only A, C, G and T.")

    if not 0 <= maximum_mismatches <= 20:
        raise ValueError("Mismatch limit must be between 0 and 20.")

    if chunk_size < 23:
        raise ValueError("Chunk size must be at least 23.")

    if repeats < 1:
        raise ValueError("Repeats must be at least 1.")

    if max_batch_starts < 1:
        raise ValueError("Batch size must be at least 1.")

    if max_chunks < 1:
        raise ValueError("Maximum chunks must be at least 1.")

    fasta_path = Path(fasta_path)

    if not fasta_path.is_file():
        raise FileNotFoundError(
            f"FASTA file not found: {fasta_path}"
        )

    backend_status = get_backend_status()

    print("\nGeneWeaver - CPU vs GPU Benchmark")
    print("=" * 60)
    print(f"FASTA file       : {fasta_path}")
    print(f"Guide sequence   : {guide_text}")
    print(f"Mismatch limit   : {maximum_mismatches}")
    print(f"Repeats          : {repeats}")
    print(f"CUDA available   : {backend_status['cuda_available']}")
    print(f"CUDA simulator   : {backend_status['simulator_enabled']}")
    print("=" * 60)

    parser = GenomicParser(
        fasta_path=str(fasta_path),
        chunk_size=chunk_size
    )

    chunk_results = []

    for chunk in parser.parse_chunks():
        result = benchmark_chunk(
            chunk=chunk,
            guide=guide_text.encode("ascii"),
            maximum_mismatches=maximum_mismatches,
            repeats=repeats,
            max_batch_starts=max_batch_starts
        )

        chunk_results.append(result)

        print(
            f"Chunk {chunk.chunk_index} | "
            f"CPU: {result['cpu']['average_seconds']:.6f}s | "
            f"Backend: {result['execution_mode']} | "
            f"Validation: "
            f"{'PASS' if result['result_validation']['results_match'] else 'FAIL'}"
        )

        if len(chunk_results) >= max_chunks:
            break

    total_windows = sum(
        item["windows_processed"]
        for item in chunk_results
    )

    total_cpu_time = sum(
        item["cpu"]["average_seconds"]
        for item in chunk_results
    )

    cuda_results_available = any(
        item["execution_mode"] == "REAL_CUDA_DEVICE"
        for item in chunk_results
    )

    simulator_results_available = any(
        item["execution_mode"] == "CUDA_SIMULATOR"
        for item in chunk_results
    )

    overall_validation = (
        bool(chunk_results)
        and all(
            item["result_validation"]["results_match"]
            for item in chunk_results
        )
    )

    if cuda_results_available:
        total_backend_time = sum(
            item["selected_backend"]["average_seconds"]
            for item in chunk_results
            if item["execution_mode"] == "REAL_CUDA_DEVICE"
        )

        comparable_speedup = (
            total_cpu_time / total_backend_time
            if total_backend_time > 0
            else None
        )

    else:
        comparable_speedup = None

    if cuda_results_available:
        benchmark_mode = "REAL_CUDA_DEVICE"
    elif simulator_results_available:
        benchmark_mode = "CUDA_SIMULATOR"
    else:
        benchmark_mode = "CPU_ONLY"

    report = {
        "project": "GeneWeaver",
        "benchmark_type": "CPU and selected backend comparison",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_mode": benchmark_mode,
        "configuration": {
            "fasta_file": str(fasta_path.resolve()),
            "guide_sequence": guide_text,
            "maximum_mismatches": maximum_mismatches,
            "chunk_size": chunk_size,
            "repeats": repeats,
            "maximum_batch_starts": max_batch_starts,
            "maximum_chunks": max_chunks
        },
        "environment": backend_status,
        "summary": {
            "chunks_processed": len(chunk_results),
            "total_windows_processed": total_windows,
            "total_cpu_average_seconds": total_cpu_time,
            "cpu_throughput_windows_per_second": (
                calculate_throughput(total_windows, total_cpu_time)
            ),
            "results_match": overall_validation,
            "measured_hardware_gpu_speedup": comparable_speedup,
            "gpu_hardware_benchmark_performed": cuda_results_available
        },
        "chunk_results": chunk_results
    }

    output_path = Path("backend_benchmark_report.json")

    with output_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=4)

    print("\nBenchmark Summary")
    print("-" * 60)
    print(f"Benchmark mode       : {benchmark_mode}")
    print(f"Chunks processed     : {len(chunk_results)}")
    print(f"Windows processed    : {total_windows}")
    print(f"CPU average total    : {total_cpu_time:.6f} seconds")
    print(
        "CPU throughput       : "
        f"{report['summary']['cpu_throughput_windows_per_second']:.2f} windows/sec"
    )
    print(f"Results match        : {overall_validation}")
    print(f"Hardware GPU tested  : {cuda_results_available}")
    print(f"Report saved         : {output_path.resolve()}")
    print("-" * 60)

    return report


def main():
    argument_parser = argparse.ArgumentParser(
        description="Benchmark GeneWeaver CPU and CUDA backends."
    )

    argument_parser.add_argument("--fasta", required=True)
    argument_parser.add_argument("--guide", required=True)
    argument_parser.add_argument("--mismatches", type=int, default=3)
    argument_parser.add_argument("--chunk-size", type=int, default=1000)
    argument_parser.add_argument("--repeats", type=int, default=3)
    argument_parser.add_argument("--max-batch-starts", type=int, default=5)
    argument_parser.add_argument("--max-chunks", type=int, default=10)

    args = argument_parser.parse_args()

    report = run_benchmark(
        fasta_path=args.fasta,
        guide_text=args.guide,
        maximum_mismatches=args.mismatches,
        chunk_size=args.chunk_size,
        repeats=args.repeats,
        max_batch_starts=args.max_batch_starts,
        max_chunks=args.max_chunks
    )

    if not report["summary"]["results_match"]:
        raise SystemExit("Benchmark validation failed.")

    if report["summary"]["chunks_processed"] == 0:
        raise SystemExit("No genomic chunks were processed.")


if __name__ == "__main__":
    main()
