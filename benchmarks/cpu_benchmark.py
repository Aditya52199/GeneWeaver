"""
CPU Baseline Benchmark Suite for GeneWeaver.

Measures baseline execution time, throughput, and MCUPS (Mega Cell Updates Per Second)
for standard DNA sequence alignment algorithms in pure Python.

Establishes the reference performance metrics for CPU execution before
benchmarking CUDA GPU-accelerated and distributed Dask engines.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
import random
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple

# Add parent directory to path so script can run directly or as a module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alignment.sequence_aligner import (
    AlignmentResult,
    AlignmentType,
    SequenceAligner,
)


@dataclass
class BenchmarkMetric:
    """Performance metrics recorded for a specific sequence length configuration."""
    seq1_length: int
    seq2_length: int
    matrix_cells: int
    trials: int
    mean_time_ms: float
    std_dev_ms: float
    min_time_ms: float
    max_time_ms: float
    mcups: float  # Mega Cell Updates Per Second based on mean time
    throughput_bp_per_sec: float
    algorithm: str
    sample_score: int
    identity_percent: float


@dataclass
class BenchmarkSummary:
    """Complete summary of the CPU benchmark run."""
    timestamp: float
    python_version: str
    algorithm: str
    results: List[BenchmarkMetric]
    overall_avg_mcups: float
    peak_mcups: float


def generate_dna_sequence(length: int, gc_content: float = 0.5, seed: Optional[int] = None) -> str:
    """
    Generate a random synthetic DNA sequence with a target GC-content.

    Args:
        length: Number of nucleotides.
        gc_content: Target fraction of G and C bases (between 0.0 and 1.0).
        seed: Random seed for reproducibility.

    Returns:
        A DNA string composed of 'A', 'C', 'G', 'T'.
    """
    if seed is not None:
        random.seed(seed)

    prob_g_or_c = gc_content / 2.0
    prob_a_or_t = (1.0 - gc_content) / 2.0
    weights = [prob_a_or_t, prob_g_or_c, prob_g_or_c, prob_a_or_t]
    bases = ["A", "C", "G", "T"]

    return "".join(random.choices(bases, weights=weights, k=length))


def mutate_dna_sequence(
    sequence: str,
    mutation_rate: float = 0.05,
    insertion_rate: float = 0.01,
    deletion_rate: float = 0.01,
    seed: Optional[int] = None,
) -> str:
    """
    Introduce realistic mutations (substitutions, insertions, deletions) into a DNA sequence.

    Args:
        sequence: Original DNA sequence.
        mutation_rate: Probability of substitution per position.
        insertion_rate: Probability of insertion per position.
        deletion_rate: Probability of deletion per position.
        seed: Random seed.

    Returns:
        Mutated DNA sequence.
    """
    if seed is not None:
        random.seed(seed)

    bases = ["A", "C", "G", "T"]
    mutated = []

    for char in sequence:
        # Check for deletion
        if random.random() < deletion_rate:
            continue

        # Check for substitution
        if random.random() < mutation_rate:
            alternate_bases = [b for b in bases if b != char]
            mutated.append(random.choice(alternate_bases))
        else:
            mutated.append(char)

        # Check for insertion
        if random.random() < insertion_rate:
            mutated.append(random.choice(bases))

    return "".join(mutated) if mutated else sequence[:1]


class CPUBenchmark:
    """Runner for CPU DNA sequence alignment performance benchmarking."""

    def __init__(
        self,
        aligner: Optional[SequenceAligner] = None,
        warmup_runs: int = 1,
    ):
        self.aligner = aligner or SequenceAligner()
        self.warmup_runs = warmup_runs

    def run_single_benchmark(
        self,
        seq1: str,
        seq2: str,
        mode: AlignmentType = AlignmentType.GLOBAL,
        trials: int = 3,
    ) -> BenchmarkMetric:
        """
        Run repeated alignments on a pair of sequences and record timing statistics.

        Args:
            seq1: First sequence.
            seq2: Second sequence.
            mode: Alignment algorithm mode.
            trials: Number of timed runs.

        Returns:
            BenchmarkMetric dataclass.
        """
        # Warmup
        for _ in range(self.warmup_runs):
            self.aligner.align(seq1, seq2, mode=mode)

        runtimes_seconds: List[float] = []
        sample_result: Optional[AlignmentResult] = None

        for _ in range(trials):
            start = time.perf_counter()
            result = self.aligner.align(seq1, seq2, mode=mode)
            elapsed = time.perf_counter() - start
            runtimes_seconds.append(elapsed)
            sample_result = result

        assert sample_result is not None

        runtimes_ms = [t * 1000.0 for t in runtimes_seconds]
        mean_ms = statistics.mean(runtimes_ms)
        std_dev_ms = statistics.stdev(runtimes_ms) if len(runtimes_ms) > 1 else 0.0
        min_ms = min(runtimes_ms)
        max_ms = max(runtimes_ms)

        mean_seconds = mean_ms / 1000.0
        cells = (len(seq1) + 1) * (len(seq2) + 1)
        mcups = (cells / (mean_seconds * 1e6)) if mean_seconds > 0 else 0.0

        total_bp = len(seq1) + len(seq2)
        throughput_bp = (total_bp / mean_seconds) if mean_seconds > 0 else 0.0

        return BenchmarkMetric(
            seq1_length=len(seq1),
            seq2_length=len(seq2),
            matrix_cells=cells,
            trials=trials,
            mean_time_ms=round(mean_ms, 3),
            std_dev_ms=round(std_dev_ms, 3),
            min_time_ms=round(min_ms, 3),
            max_time_ms=round(max_ms, 3),
            mcups=round(mcups, 4),
            throughput_bp_per_sec=round(throughput_bp, 2),
            algorithm=mode.value if isinstance(mode, AlignmentType) else str(mode),
            sample_score=sample_result.score,
            identity_percent=sample_result.identity_percent,
        )

    def run_scaling_suite(
        self,
        sequence_lengths: List[int],
        mode: AlignmentType = AlignmentType.GLOBAL,
        trials: int = 3,
        gc_content: float = 0.5,
        mutation_rate: float = 0.05,
    ) -> BenchmarkSummary:
        """
        Run benchmarks across a range of sequence lengths.

        Args:
            sequence_lengths: List of sequence lengths (e.g. [50, 100, 200, 500, 1000]).
            mode: AlignmentType to test.
            trials: Number of trials per length.
            gc_content: GC ratio for random sequence generation.
            mutation_rate: Mutation probability.

        Returns:
            BenchmarkSummary.
        """
        results: List[BenchmarkMetric] = []

        for length in sequence_lengths:
            seq1 = generate_dna_sequence(length, gc_content=gc_content, seed=42 + length)
            seq2 = mutate_dna_sequence(seq1, mutation_rate=mutation_rate, seed=100 + length)

            metric = self.run_single_benchmark(seq1, seq2, mode=mode, trials=trials)
            results.append(metric)

        mcups_list = [r.mcups for r in results]
        overall_avg_mcups = round(statistics.mean(mcups_list), 4) if mcups_list else 0.0
        peak_mcups = max(mcups_list) if mcups_list else 0.0

        return BenchmarkSummary(
            timestamp=time.time(),
            python_version=sys.version.split()[0],
            algorithm=mode.value if isinstance(mode, AlignmentType) else str(mode),
            results=results,
            overall_avg_mcups=overall_avg_mcups,
            peak_mcups=peak_mcups,
        )

    def run_crispr_guide_benchmark(
        self,
        target_lengths: List[int] = [23, 100, 500, 1000, 2000],
        guide_length: int = 20,
        trials: int = 5,
    ) -> BenchmarkSummary:
        """
        Benchmark short CRISPR guide RNA (e.g. 20bp) aligned against varying genomic window lengths.
        """
        results: List[BenchmarkMetric] = []
        guide_seq = generate_dna_sequence(guide_length, gc_content=0.5, seed=2026)

        for target_len in target_lengths:
            target_seq = generate_dna_sequence(target_len, gc_content=0.5, seed=42 + target_len)
            # Embed guide with 1-2 mismatches inside target if target is large enough
            if target_len >= guide_length:
                insert_idx = target_len // 2
                mutated_guide = mutate_dna_sequence(guide_seq, mutation_rate=0.1, seed=77)
                target_seq = (
                    target_seq[:insert_idx]
                    + mutated_guide
                    + target_seq[insert_idx + len(mutated_guide) :]
                )[:target_len]

            metric = self.run_single_benchmark(
                guide_seq,
                target_seq,
                mode=AlignmentType.SEMI_GLOBAL,
                trials=trials,
            )
            results.append(metric)

        mcups_list = [r.mcups for r in results]
        overall_avg_mcups = round(statistics.mean(mcups_list), 4) if mcups_list else 0.0
        peak_mcups = max(mcups_list) if mcups_list else 0.0

        return BenchmarkSummary(
            timestamp=time.time(),
            python_version=sys.version.split()[0],
            algorithm="crispr_semi_global",
            results=results,
            overall_avg_mcups=overall_avg_mcups,
            peak_mcups=peak_mcups,
        )


def format_benchmark_table(summary: BenchmarkSummary) -> str:
    """Format benchmark results into a clean ASCII table."""
    header = (
        f"\n========================================================================================\n"
        f" GeneWeaver CPU Baseline Benchmark: {summary.algorithm.upper()}\n"
        f" Python: {summary.python_version} | Peak MCUPS: {summary.peak_mcups:.4f} | Avg MCUPS: {summary.overall_avg_mcups:.4f}\n"
        f"========================================================================================\n"
        f" {'Seq1 (bp)':<10} | {'Seq2 (bp)':<10} | {'Matrix Cells':<14} | {'Avg Time (ms)':<14} | {'Std Dev':<10} | {'MCUPS':<10} | {'Throughput (bp/s)':<18}\n"
        f"----------------------------------------------------------------------------------------"
    )
    rows = []
    for r in summary.results:
        row = (
            f" {r.seq1_length:<10} | {r.seq2_length:<10} | {r.matrix_cells:<14,d} | "
            f"{r.mean_time_ms:<14.3f} | {r.std_dev_ms:<10.3f} | {r.mcups:<10.4f} | "
            f"{r.throughput_bp_per_sec:<18,.1f}"
        )
        rows.append(row)

    footer = "========================================================================================"
    return "\n".join([header] + rows + [footer])


def save_benchmark_report(summary: BenchmarkSummary, filepath: str) -> None:
    """Save benchmark summary to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)


def main():
    """Command-line interface to execute CPU benchmarks."""
    parser = argparse.ArgumentParser(
        description="GeneWeaver CPU Baseline Performance Benchmark"
    )
    parser.add_argument(
        "--algorithm",
        choices=["global", "local", "semi_global", "crispr", "all"],
        default="all",
        help="Algorithm mode to benchmark (default: all)",
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[50, 100, 200, 400, 800],
        help="Sequence lengths for scaling test (default: 50 100 200 400 800)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials per length (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/baseline_metrics.json",
        help="Output JSON file for benchmark metrics",
    )

    args = parser.parse_args()
    benchmark_runner = CPUBenchmark()

    summaries = []

    if args.algorithm in ["global", "all"]:
        print(f"\n[Running Global Alignment (Needleman-Wunsch) Benchmark...]")
        summary_nw = benchmark_runner.run_scaling_suite(
            sequence_lengths=args.lengths,
            mode=AlignmentType.GLOBAL,
            trials=args.trials,
        )
        print(format_benchmark_table(summary_nw))
        summaries.append(summary_nw)

    if args.algorithm in ["local", "all"]:
        print(f"\n[Running Local Alignment (Smith-Waterman) Benchmark...]")
        summary_sw = benchmark_runner.run_scaling_suite(
            sequence_lengths=args.lengths,
            mode=AlignmentType.LOCAL,
            trials=args.trials,
        )
        print(format_benchmark_table(summary_sw))
        summaries.append(summary_sw)

    if args.algorithm in ["crispr", "all"]:
        print(f"\n[Running CRISPR Guide Semi-Global Benchmark...]")
        summary_crispr = benchmark_runner.run_crispr_guide_benchmark(
            target_lengths=[23, 100, 500, 1000],
            guide_length=20,
            trials=args.trials,
        )
        print(format_benchmark_table(summary_crispr))
        summaries.append(summary_crispr)

    # Save summary
    if summaries:
        all_data = {
            "benchmarks": [asdict(s) for s in summaries],
            "system_info": {
                "python": sys.version,
                "platform": sys.platform,
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2)
        print(f"\n[+] Baseline benchmark metrics saved to: {args.output}\n")


if __name__ == "__main__":
    main()

