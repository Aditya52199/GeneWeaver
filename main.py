"""
GeneWeaver: GPU-Accelerated CRISPR Alignment Engine
Main CLI entry point and data pipeline runner.
"""

import argparse
import sys
import time

from genomic_parser import GenomePipeline
from scripts.generate_test_data import create_mock_genome_fasta


def run_pipeline_demo(
    fasta_path: str = "data/mock_human_genome.fasta",
    chunk_size: int = 10_000,
    overlap: int = 100,
):
    """Run demonstration of BioPython genome ingestion and array chunking."""
    print("=" * 70)
    print(" GeneWeaver: Genomic Ingestion and Array Chunking Pipeline")
    print("=" * 70)

    start_time = time.perf_counter()
    pipeline = GenomePipeline(
        fasta_path=fasta_path,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    print(f"\n[1] Parsing FASTA using BioPython: {fasta_path}")
    summary = pipeline.get_summary()

    print(f"    - Total Genome Length : {summary.total_length:,} bp")
    print(f"    - Chromosomes Loaded  : {summary.chromosome_count}")
    print(f"    - Overall GC Content  : {summary.average_gc_content:.2f}%")
    print(f"    - Ambiguous (N) Bases : {summary.total_ambiguous_bases:,}")

    print("\n[2] Chromosome Overview:")
    for chrom in summary.chromosomes:
        print(
            f"    * {chrom.name:<6} | Length: {chrom.length:>7,d} bp | "
            f"GC: {chrom.gc_content:>5.2f}% | N-bases: {chrom.ambiguous_percentage:>4.2f}%"
        )

    print(f"\n[3] Chunking into NumPy Arrays (chunk_size={chunk_size:,} bp, overlap={overlap:,} bp)...")
    chunks = pipeline.process_all_chunks()

    elapsed = time.perf_counter() - start_time
    print(f"    - Total Chunks Created: {len(chunks):,}")
    print(f"    - Pipeline Time       : {elapsed * 1000.0:.2f} ms")
    print(f"    - Processing Rate     : {summary.total_length / elapsed:,.0f} bp/sec")

    print("\n[4] Sample Chunk Arrays (Ready for GPU Memory Transfer):")
    sample_indices = [0, len(chunks) // 2, len(chunks) - 1]
    for idx in sample_indices:
        c = chunks[idx]
        preview_bytes = c.array[:16]
        preview_str = c.as_str()[:16]
        print(f"    Chunk #{c.chunk_id} [{c.chromosome}:{c.start_pos:,}-{c.end_pos:,}] (len={c.length:,} bp, GC={c.gc_content:.1f}%):")
        print(f"      NumPy array dtype : {c.array.dtype}, shape: {c.array.shape}")
        print(f"      Byte values (hex) : {' '.join(f'{b:02X}' for b in preview_bytes)}...")
        print(f"      ASCII sequence    : {preview_str}...")

    print("\n" + "=" * 70)
    print(" Pipeline execution completed successfully.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="GeneWeaver Genomic Pipeline")
    parser.add_argument(
        "--fasta",
        type=str,
        default="data/mock_human_genome.fasta",
        help="Path to genome FASTA file (.fasta or .fasta.gz)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Chunk size in nucleotides (default: 10,000)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Overlap size between chunks (default: 100)",
    )
    parser.add_argument(
        "--dashboard", "--tui",
        action="store_true",
        help="Launch the interactive Textual terminal dashboard",
    )
    args = parser.parse_args()

    if args.dashboard:
        from dashboard import run_dashboard
        run_dashboard()
    else:
        run_pipeline_demo(
            fasta_path=args.fasta,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )


if __name__ == "__main__":
    main()
