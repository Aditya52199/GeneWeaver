"""
Mock Human Genome FASTA Generator for GeneWeaver.

Generates realistic mock human genome FASTA files with:
- Multiple chromosomes (e.g., chr1, chr2, chr21, chrX)
- Human-accurate GC-content (~41%)
- Soft-masked repetitive elements (lowercase nucleotides)
- Centromeric/unsequenced N-gaps
- Natural SpCas9 PAM sites (NGG)
"""

import argparse
import gzip
import os
import random
import sys
from typing import Dict, List, Optional


def generate_human_like_sequence(
    length: int,
    gc_content: float = 0.41,
    repeat_fraction: float = 0.05,
    n_gap_fraction: float = 0.02,
    seed: Optional[int] = None,
) -> str:
    """Generate a realistic synthetic chromosome sequence."""
    if seed is not None:
        random.seed(seed)

    prob_g_or_c = gc_content / 2.0
    prob_a_or_t = (1.0 - gc_content) / 2.0
    weights = [prob_a_or_t, prob_g_or_c, prob_g_or_c, prob_a_or_t]
    bases = ["A", "C", "G", "T"]

    seq_chars = random.choices(bases, weights=weights, k=length)

    # Insert a few stretches of unsequenced N-gaps (e.g., centromeric/telomeric scaffolds)
    num_n_gaps = max(1, int(length * n_gap_fraction / 50))
    for _ in range(num_n_gaps):
        gap_len = random.randint(20, min(100, length))
        gap_start = random.randint(0, max(0, length - gap_len))
        for i in range(gap_start, gap_start + gap_len):
            seq_chars[i] = "N"

    # Soft-mask some repeat regions (lowercase acgt)
    num_repeats = max(1, int(length * repeat_fraction / 40))
    for _ in range(num_repeats):
        rep_len = random.randint(15, min(60, length))
        rep_start = random.randint(0, max(0, length - rep_len))
        for i in range(rep_start, rep_start + rep_len):
            seq_chars[i] = seq_chars[i].lower()

    return "".join(seq_chars)


def create_mock_genome_fasta(
    output_path: str = "data/mock_human_genome.fasta",
    chromosomes: Optional[Dict[str, int]] = None,
    compress_gzip: bool = False,
    seed: int = 42,
) -> str:
    """
    Generate and write a multi-chromosome mock human genome FASTA file.

    Args:
        output_path: Target path for the FASTA file.
        chromosomes: Mapping of chromosome name to length in bp.
        compress_gzip: Whether to write as a .fasta.gz file.
        seed: Random seed.

    Returns:
        Absolute path to the created file.
    """
    if chromosomes is None:
        chromosomes = {
            "chr1": 60_000,
            "chr2": 45_000,
            "chr21": 30_000,
            "chrX": 25_000,
        }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if compress_gzip and not output_path.endswith(".gz"):
        output_path += ".gz"

    open_fn = gzip.open if output_path.endswith(".gz") else open
    mode = "wt" if output_path.endswith(".gz") else "w"

    line_width = 80

    with open_fn(output_path, mode, encoding="utf-8") as f:
        for idx, (chr_name, chr_len) in enumerate(chromosomes.items()):
            header = f">{chr_name} Homo sapiens chromosome {chr_name}, mock GRCh38 assembly\n"
            f.write(header)

            seq = generate_human_like_sequence(
                length=chr_len,
                gc_content=0.41,
                seed=seed + idx * 100,
            )

            for i in range(0, len(seq), line_width):
                f.write(seq[i : i + line_width] + "\n")

    return os.path.abspath(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate Mock Human Genome FASTA")
    parser.add_argument(
        "--output",
        type=str,
        default="data/mock_human_genome.fasta",
        help="Output FASTA file path (default: data/mock_human_genome.fasta)",
    )
    parser.add_argument(
        "--gzip",
        action="store_true",
        help="Compress output with gzip (.fasta.gz)",
    )
    parser.add_argument(
        "--size-scale",
        type=float,
        default=1.0,
        help="Multiplier for chromosome lengths (default: 1.0)",
    )

    args = parser.parse_args()

    chroms = {
        "chr1": int(60_000 * args.size_scale),
        "chr2": int(45_000 * args.size_scale),
        "chr21": int(30_000 * args.size_scale),
        "chrX": int(25_000 * args.size_scale),
    }

    out_file = create_mock_genome_fasta(
        output_path=args.output,
        chromosomes=chroms,
        compress_gzip=args.gzip,
    )

    print(f"[+] Successfully generated mock human genome FASTA at: {out_file}")
    total_bp = sum(chroms.values())
    print(f"[+] Chromosomes: {list(chroms.keys())} | Total length: {total_bp:,} bp")


if __name__ == "__main__":
    main()

