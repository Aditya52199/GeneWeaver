
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from geneweaver import (
    GenomicParser,
    GUIDE_LENGTH,
    scan_chunk,
)

from cuda_backend import scan_chunk_accelerated, get_backend_status


def validate_guide(guide_text):
    guide_text = guide_text.upper()

    if len(guide_text) != GUIDE_LENGTH:
        raise ValueError("Guide must contain exactly 20 bases.")

    if any(base not in "ACGT" for base in guide_text):
        raise ValueError("Guide must contain only A, C, G and T.")

    return guide_text.encode("ascii")


def candidate_signature(candidate):
    """
    Select fields that must match between CPU and CUDA results.
    """
    return (
        candidate["chromosome"],
        candidate["start_position"],
        candidate["end_position"],
        candidate["strand"],
        candidate["guide"],
        candidate["target"],
        candidate["pam"],
        candidate["mismatch_count"],
        candidate["mismatch_positions"],
        candidate["seed_mismatches"],
        candidate["gc_content"],
        candidate["candidate_score"],
        candidate["priority"],
    )


def compare_candidates(cpu_candidates, accelerated_candidates):
    """
    Compare candidate collections without depending on their order.
    Duplicate candidates are preserved and compared.
    """

    cpu_signatures = sorted(
        candidate_signature(item)
        for item in cpu_candidates
    )

    accelerated_signatures = sorted(
        candidate_signature(item)
        for item in accelerated_candidates
    )

    missing_from_accelerated = []
    extra_in_accelerated = []

    cpu_remaining = list(cpu_signatures)
    accelerated_remaining = list(accelerated_signatures)

    for signature in cpu_signatures:
        if signature in accelerated_remaining:
            accelerated_remaining.remove(signature)
        else:
            missing_from_accelerated.append(signature)

    for signature in accelerated_signatures:
        if signature in cpu_remaining:
            cpu_remaining.remove(signature)
        else:
            extra_in_accelerated.append(signature)

    return {
        "matches": (
            not missing_from_accelerated
            and not extra_in_accelerated
        ),
        "cpu_candidate_count": len(cpu_candidates),
        "accelerated_candidate_count": len(accelerated_candidates),
        "missing_count": len(missing_from_accelerated),
        "extra_count": len(extra_in_accelerated),
        "missing_candidates": [
            list(item) for item in missing_from_accelerated
        ],
        "extra_candidates": [
            list(item) for item in extra_in_accelerated
        ]
    }


def validate_chunk(chunk, guide, maximum_mismatches, max_batch_starts):
    """
    Compare the reference CPU scanner with the selected backend.
    """

    cpu_candidates, cpu_stats = scan_chunk(
        chunk,
        guide,
        maximum_mismatches
    )

    accelerated_result = scan_chunk_accelerated(
        chunk,
        guide,
        maximum_mismatches,
        max_batch_starts=max_batch_starts
    )

    accelerated_candidates = accelerated_result["candidates"]
    accelerated_stats = accelerated_result["statistics"]

    comparison = compare_candidates(
        cpu_candidates,
        accelerated_candidates
    )

    pam_match = (
        cpu_stats["pam_sites"]
        == accelerated_stats["pam_sites"]
    )

    comparison["pam_count_matches"] = pam_match

    comparison["matches"] = (
        comparison["matches"] and pam_match
    )

    return {
        "chunk_index": chunk.chunk_index,
        "chromosome": chunk.chromosome,
        "backend": accelerated_result["backend"],
        "validation_passed": comparison["matches"],
        "cpu": {
            "pam_sites": cpu_stats["pam_sites"],
            "candidate_count": len(cpu_candidates)
        },
        "accelerated": {
            "pam_sites": accelerated_stats["pam_sites"],
            "candidate_count": len(accelerated_candidates)
        },
        "comparison": comparison
    }


def run_validation(
    fasta_path,
    guide_text,
    maximum_mismatches,
    chunk_size,
    max_batch_starts
):
    guide = validate_guide(guide_text)

    if not 0 <= maximum_mismatches <= GUIDE_LENGTH:
        raise ValueError("Mismatch limit must be between 0 and 20.")

    if chunk_size < 23:
        raise ValueError("Chunk size must be at least 23.")

    if max_batch_starts < 1:
        raise ValueError("Maximum batch size must be at least 1.")

    if not Path(fasta_path).is_file():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    backend_status = get_backend_status()

    validation_results = []
    total_chunks = 0
    passed_chunks = 0
    failed_chunks = 0
    actual_backend = None

    parser = GenomicParser(
        fasta_path=fasta_path,
        chunk_size=chunk_size
    )

    print("\nGeneWeaver - Alignment Validation")
    print("=" * 60)
    print(f"FASTA file        : {fasta_path}")
    print(f"Guide sequence    : {guide_text.upper()}")
    print(f"Maximum mismatches: {maximum_mismatches}")
    print(f"CUDA available    : {backend_status['cuda_available']}")
    print("=" * 60)

    for chunk in parser.parse_chunks():
        result = validate_chunk(
            chunk=chunk,
            guide=guide,
            maximum_mismatches=maximum_mismatches,
            max_batch_starts=max_batch_starts
        )

        validation_results.append(result)

        total_chunks += 1
        actual_backend = result["backend"]

        if result["validation_passed"]:
            passed_chunks += 1
            status = "PASS"
        else:
            failed_chunks += 1
            status = "FAIL"

        print(
            f"Chunk {chunk.chunk_index:>5} | "
            f"Backend: {result['backend']:<4} | "
            f"Candidates: {result['cpu']['candidate_count']:>5} | "
            f"Validation: {status}"
        )

    if actual_backend == "CUDA":
        validation_mode = "CUDA_KERNEL_VALIDATION"
    elif total_chunks == 0:
        validation_mode = "NO_CHUNKS_PROCESSED"
    else:
        validation_mode = "CPU_FALLBACK_VALIDATION"

    report = {
        "project": "GeneWeaver",
        "validation_type": "CPU versus selected alignment backend",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "fasta_file": str(Path(fasta_path).resolve()),
            "guide_sequence": guide_text.upper(),
            "maximum_mismatches": maximum_mismatches,
            "chunk_size": chunk_size,
            "maximum_batch_starts": max_batch_starts
        },
        "environment": backend_status,
        "validation_mode": validation_mode,
        "summary": {
            "total_chunks": total_chunks,
            "passed_chunks": passed_chunks,
            "failed_chunks": failed_chunks,
            "overall_passed": failed_chunks == 0 and total_chunks > 0
        },
        "chunk_results": validation_results
    }

    output_path = Path("validation_report.json")

    with output_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=4)

    print("\nValidation Summary")
    print("-" * 60)
    print(f"Chunks processed : {total_chunks}")
    print(f"Passed           : {passed_chunks}")
    print(f"Failed           : {failed_chunks}")
    print(f"Validation mode  : {validation_mode}")
    print(f"Overall result   : {report['summary']['overall_passed']}")
    print(f"Report saved     : {output_path.resolve()}")
    print("-" * 60)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate GeneWeaver alignment results."
    )

    parser.add_argument("--fasta", required=True)
    parser.add_argument("--guide", required=True)
    parser.add_argument("--mismatches", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--max-batch-starts", type=int, default=5)

    args = parser.parse_args()

    report = run_validation(
        fasta_path=args.fasta,
        guide_text=args.guide,
        maximum_mismatches=args.mismatches,
        chunk_size=args.chunk_size,
        max_batch_starts=args.max_batch_starts
    )

    if not report["summary"]["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
