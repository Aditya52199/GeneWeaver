
import argparse
import gc
import json
import statistics
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

from geneweaver import (
    GenomicParser,
    scan_chunk,
    GUIDE_LENGTH,
)


def validate_guide(guide_text):
    guide_text = guide_text.upper()

    if len(guide_text) != GUIDE_LENGTH:
        raise ValueError("Guide sequence must contain exactly 20 bases.")

    if any(base not in "ACGT" for base in guide_text):
        raise ValueError("Guide sequence must contain only A, C, G and T.")

    return guide_text.encode("ascii")


def run_single_scan(fasta_path, guide, mismatches, chunk_size):
    parser = GenomicParser(
        fasta_path=fasta_path,
        chunk_size=chunk_size
    )

    chunks_processed = 0
    valid_windows_per_strand = 0
    pam_sites = 0
    candidates = 0

    for chunk in parser.parse_chunks():
        found, stats = scan_chunk(
            chunk=chunk,
            guide=guide,
            maximum_mismatches=mismatches
        )

        chunks_processed += 1
        valid_windows_per_strand += chunk.valid_starts
        pam_sites += stats["pam_sites"]
        candidates += len(found)

        # Do not retain candidates: this benchmark measures scanning,
        # not storage of all candidate records.

    total_windows = valid_windows_per_strand * 2

    return {
        "chunks_processed": chunks_processed,
        "windows_processed": total_windows,
        "pam_sites": pam_sites,
        "candidates": candidates
    }


def benchmark_cpu(
    fasta_path,
    guide_text,
    mismatches,
    chunk_size,
    repeats
):
    guide = validate_guide(guide_text)

    if repeats < 1:
        raise ValueError("Repeats must be at least 1.")

    if not Path(fasta_path).is_file():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    timings = []
    final_scan_stats = None

    print("\nGeneWeaver - CPU Benchmark")
    print("=" * 55)
    print(f"FASTA file       : {fasta_path}")
    print(f"Guide sequence   : {guide_text.upper()}")
    print(f"Mismatch limit   : {mismatches}")
    print(f"Chunk size       : {chunk_size:,}")
    print(f"Benchmark runs   : {repeats}")
    print("=" * 55)

    # Timing runs without tracemalloc, to reduce profiling overhead.
    for run_number in range(1, repeats + 1):
        gc.collect()

        start_time = time.perf_counter()

        scan_stats = run_single_scan(
            fasta_path=fasta_path,
            guide=guide,
            mismatches=mismatches,
            chunk_size=chunk_size
        )

        elapsed = time.perf_counter() - start_time
        timings.append(elapsed)
        final_scan_stats = scan_stats

        print(
            f"Run {run_number:>2}: "
            f"{elapsed:.4f} seconds | "
            f"{scan_stats['windows_processed']:,} windows | "
            f"{scan_stats['candidates']:,} candidates"
        )

    # Run one additional scan with tracemalloc enabled.
    # This measures Python-tracked allocations, not total system RAM.
    gc.collect()
    tracemalloc.start()

    try:
        memory_scan_stats = run_single_scan(
            fasta_path=fasta_path,
            guide=guide,
            mismatches=mismatches,
            chunk_size=chunk_size
        )

        current_memory, peak_memory = tracemalloc.get_traced_memory()

    finally:
        tracemalloc.stop()

    average_time = statistics.mean(timings)
    minimum_time = min(timings)
    maximum_time = max(timings)

    windows_processed = final_scan_stats["windows_processed"]

    throughput = (
        windows_processed / average_time
        if average_time > 0
        else 0.0
    )

    report = {
        "project": "GeneWeaver",
        "benchmark_type": "CPU",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "fasta_file": str(Path(fasta_path).resolve()),
            "guide_sequence": guide_text.upper(),
            "maximum_mismatches": mismatches,
            "chunk_size": chunk_size,
            "repeats": repeats
        },
        "scan_results": final_scan_stats,
        "timing_seconds": {
            "individual_runs": [round(value, 6) for value in timings],
            "average": round(average_time, 6),
            "minimum": round(minimum_time, 6),
            "maximum": round(maximum_time, 6)
        },
        "performance": {
            "windows_per_second": round(throughput, 2)
        },
        "python_memory_profiling": {
            "current_traced_memory_mb": round(
                current_memory / (1024 * 1024), 4
            ),
            "peak_traced_memory_mb": round(
                peak_memory / (1024 * 1024), 4
            ),
            "measurement": "Python allocations tracked by tracemalloc"
        }
    }

    output_path = Path("cpu_benchmark_report.json")

    with output_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=4)

    print("\nBenchmark Summary")
    print("-" * 55)
    print(f"Chunks processed       : {final_scan_stats['chunks_processed']:,}")
    print(f"Windows processed     : {windows_processed:,}")
    print(f"PAM sites found       : {final_scan_stats['pam_sites']:,}")
    print(f"Candidates found      : {final_scan_stats['candidates']:,}")
    print(f"Average time          : {average_time:.4f} seconds")
    print(f"Minimum time          : {minimum_time:.4f} seconds")
    print(f"Maximum time          : {maximum_time:.4f} seconds")
    print(f"Throughput            : {throughput:,.2f} windows/second")
    print(
        "Peak Python memory   : "
        f"{peak_memory / (1024 * 1024):.4f} MB"
    )
    print(f"Report saved          : {output_path.resolve()}")
    print("-" * 55)

    # Confirm that the profiling pass produced the same scan totals.
    if memory_scan_stats != final_scan_stats:
        print("WARNING: Profiling scan totals differ from timing scan totals.")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the GeneWeaver CPU alignment engine."
    )

    parser.add_argument(
        "--fasta",
        required=True,
        help="Input FASTA file"
    )

    parser.add_argument(
        "--guide",
        required=True,
        help="20-base DNA guide sequence"
    )

    parser.add_argument(
        "--mismatches",
        type=int,
        default=3,
        help="Maximum allowed mismatches"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="FASTA chunk size"
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed benchmark runs"
    )

    args = parser.parse_args()

    if not 0 <= args.mismatches <= GUIDE_LENGTH:
        parser.error("Mismatch limit must be between 0 and 20.")

    if args.chunk_size < 23:
        parser.error("Chunk size must be at least 23.")

    if args.repeats < 1:
        parser.error("Repeats must be at least 1.")

    benchmark_cpu(
        fasta_path=args.fasta,
        guide_text=args.guide,
        mismatches=args.mismatches,
        chunk_size=args.chunk_size,
        repeats=args.repeats
    )


if __name__ == "__main__":
    main()
