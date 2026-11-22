
"""
GeneWeaver - Commit 10
Batch processing for genomic CRISPR alignment.

Uses the existing CPU or CUDA alignment backend.
CUDA simulation is not treated as real GPU hardware execution.
"""

import argparse
import inspect
import json
import os
import time
from pathlib import Path

from geneweaver import GenomicParser, scan_chunk
from cuda_backend import scan_chunk_accelerated, get_backend_status


def select_backend(requested_backend="auto"):
    """
    Select the alignment backend.

    Options:
        auto        - use CUDA if available, otherwise CPU
        cpu         - always use CPU
        accelerated - use the existing accelerated backend
    """
    status = get_backend_status()

    cuda_available = bool(status.get("cuda_available", False))
    simulator_enabled = bool(
        status.get("simulator_enabled", False)
    )

    if requested_backend == "cpu":
        return scan_chunk, "CPU"

    if requested_backend == "accelerated":
        if cuda_available:
            return scan_chunk_accelerated, "CUDA"
        if simulator_enabled:
            return scan_chunk_accelerated, "CUDA_SIMULATOR"

        raise RuntimeError(
            "CUDA hardware is unavailable and CUDA simulator is disabled."
        )

    if requested_backend != "auto":
        raise ValueError(
            "Backend must be auto, cpu, or accelerated."
        )

    if cuda_available:
        return scan_chunk_accelerated, "CUDA"

    if simulator_enabled:
        return scan_chunk_accelerated, "CUDA_SIMULATOR"

    return scan_chunk, "CPU"



def execute_chunk(
    backend_function,
    chunk,
    guide_bytes,
    maximum_mismatches,
    max_batch_starts
):
    """
    Execute one genomic chunk.

    Pass max_batch_starts only when it is explicitly
    supported by the backend function signature.
    """
    try:
        parameters = inspect.signature(
            backend_function
        ).parameters
    except (TypeError, ValueError):
        parameters = {}

    supports_batch_argument = (
        "max_batch_starts" in parameters
    )

    if supports_batch_argument:
        return backend_function(
            chunk,
            guide_bytes,
            maximum_mismatches,
            max_batch_starts=max_batch_starts
        )

    return backend_function(
        chunk,
        guide_bytes,
        maximum_mismatches
    )



def aggregate_statistics(total_stats, current_stats):
    """Add numeric statistics returned by each processed chunk."""
    if not isinstance(current_stats, dict):
        return

    for key, value in current_stats.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total_stats[key] = total_stats.get(key, 0) + value


def process_chunks(
    chunks,
    guide_bytes,
    maximum_mismatches=3,
    max_batch_starts=10000,
    backend="auto"
):
    """
    Process an iterable of genomic chunks.

    Returns a dictionary containing candidates, statistics,
    selected backend, and processing time.
    """
    if maximum_mismatches < 0:
        raise ValueError("Maximum mismatches cannot be negative.")

    if max_batch_starts < 1:
        raise ValueError("max_batch_starts must be at least 1.")

    backend_function, backend_name = select_backend(backend)

    all_candidates = []
    total_stats = {}
    chunks_processed = 0

    start_time = time.perf_counter()

    for chunk in chunks:
        result = execute_chunk(
            backend_function,
            chunk,
            guide_bytes,
            maximum_mismatches,
            max_batch_starts
        )

        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError(
                "Alignment backend must return (candidates, statistics)."
            )

        candidates, stats = result

        if candidates is None:
            candidates = []

        all_candidates.extend(candidates)
        aggregate_statistics(total_stats, stats)

        chunks_processed += 1

    elapsed_time = time.perf_counter() - start_time

    return {
        "candidates": all_candidates,
        "statistics": total_stats,
        "chunks_processed": chunks_processed,
        "candidate_count": len(all_candidates),
        "backend": backend_name,
        "processing_time_seconds": elapsed_time
    }


def validate_guide(guide):
    """Validate a 20-nucleotide DNA guide sequence."""
    guide = guide.strip().upper()

    if len(guide) != 20:
        raise ValueError("Guide RNA sequence must contain exactly 20 bases.")

    if any(base not in "ACGT" for base in guide):
        raise ValueError("Guide RNA must contain only A, C, G, and T.")

    return guide.encode("ascii")


def candidate_to_dict(candidate):
    """Convert a candidate object into JSON-compatible data."""
    if isinstance(candidate, dict):
        return candidate

    if hasattr(candidate, "__dict__"):
        return candidate.__dict__

    return str(candidate)


def run_batch_processing(
    fasta_path,
    guide,
    maximum_mismatches=3,
    chunk_size=100000,
    max_batch_starts=10000,
    backend="auto"
):
    """Read a FASTA file and process all genomic chunks."""
    fasta_path = Path(fasta_path)

    if not fasta_path.is_file():
        raise FileNotFoundError(
            f"FASTA file not found: {fasta_path}"
        )

    guide_bytes = validate_guide(guide)

    parser = GenomicParser(
        str(fasta_path),
        chunk_size=chunk_size
    )

    chunks = parser.parse_chunks()

    result = process_chunks(
        chunks=chunks,
        guide_bytes=guide_bytes,
        maximum_mismatches=maximum_mismatches,
        max_batch_starts=max_batch_starts,
        backend=backend
    )

    status = get_backend_status()

    if result["backend"] == "CUDA":
        execution_note = "CUDA hardware backend selected."
    elif result["backend"] == "CUDA_SIMULATOR":
        execution_note = (
            "CUDA simulator selected. This is not real GPU hardware execution."
        )
    else:
        execution_note = (
            "CPU backend selected. No CUDA hardware execution was performed."
        )

    report = {
        "project": "GeneWeaver",
        "commit": 10,
        "operation": "Genomic batch processing",
        "input_fasta": str(fasta_path),
        "guide_sequence": guide,
        "maximum_mismatches": maximum_mismatches,
        "chunk_size": chunk_size,
        "max_batch_starts": max_batch_starts,
        "backend": result["backend"],
        "execution_note": execution_note,
        "cuda_available": bool(status.get("cuda_available", False)),
        "chunks_processed": result["chunks_processed"],
        "candidate_count": result["candidate_count"],
        "processing_time_seconds": result["processing_time_seconds"],
        "statistics": result["statistics"],
        "candidates": [
            candidate_to_dict(candidate)
            for candidate in result["candidates"]
        ]
    }

    report_path = Path("batch_processing_report.json")

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, default=str)

    return report, report_path


def main():
    parser = argparse.ArgumentParser(
        description="GeneWeaver genomic batch processor"
    )

    parser.add_argument(
        "--fasta",
        required=True,
        help="Path to the input FASTA file"
    )

    parser.add_argument(
        "--guide",
        required=True,
        help="20-base guide RNA sequence"
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
        help="Number of bases per genomic chunk"
    )

    parser.add_argument(
        "--max-batch-starts",
        type=int,
        default=10000,
        help="Maximum alignment starts per backend batch, if supported"
    )

    parser.add_argument(
        "--backend",
        choices=["auto", "cpu", "accelerated"],
        default="auto",
        help="Select the execution backend"
    )

    args = parser.parse_args()

    if args.mismatches < 0:
        parser.error("--mismatches cannot be negative.")

    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1.")

    if args.max_batch_starts < 1:
        parser.error("--max-batch-starts must be at least 1.")

    try:
        report, report_path = run_batch_processing(
            fasta_path=args.fasta,
            guide=args.guide,
            maximum_mismatches=args.mismatches,
            chunk_size=args.chunk_size,
            max_batch_starts=args.max_batch_starts,
            backend=args.backend
        )

        print("\n========== GENEWEAVER BATCH PROCESSING ==========")
        print("Backend             :", report["backend"])
        print("Chunks processed   :", report["chunks_processed"])
        print("Candidates detected:", report["candidate_count"])
        print(
            "Processing time    :",
            f'{report["processing_time_seconds"]:.6f} seconds'
        )
        print("Execution note     :", report["execution_note"])
        print("Report saved to    :", report_path.resolve())
        print("=================================================\n")

    except Exception as error:
        parser.exit(
            1,
            f"\nBatch processing failed: {error}\n"
        )


if __name__ == "__main__":
    main()
