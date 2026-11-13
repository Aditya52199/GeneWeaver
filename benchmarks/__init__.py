"""
GeneWeaver Benchmarks Package.

Provides CPU baseline and accelerated comparison benchmarks.
"""

from benchmarks.cpu_benchmark import (
    BenchmarkMetric,
    BenchmarkSummary,
    CPUBenchmark,
    generate_dna_sequence,
    mutate_dna_sequence,
)

__all__ = [
    "BenchmarkMetric",
    "BenchmarkSummary",
    "CPUBenchmark",
    "generate_dna_sequence",
    "mutate_dna_sequence",
]

