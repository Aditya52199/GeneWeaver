"""
Genomic sequence validator for GeneWeaver.

Validates FASTA records and sequence properties (length, composition, GC-content).
"""

from typing import Dict, Tuple


class SequenceValidator:
    """Validates sequence statistics, GC content, and validity for downstream alignment."""

    @staticmethod
    def calculate_composition(sequence: str) -> Dict[str, int]:
        """Count nucleotide occurrences in a DNA sequence."""
        counts = {"A": 0, "C": 0, "G": 0, "T": 0, "N": 0, "other": 0}
        for char in sequence.upper():
            if char in counts:
                counts[char] += 1
            else:
                counts["other"] += 1
        return counts

    @staticmethod
    def calculate_gc_content(sequence: str) -> float:
        """Calculate GC content percentage (excluding N/other bases from denominator)."""
        if not sequence:
            return 0.0

        seq_upper = sequence.upper()
        gc_count = seq_upper.count("G") + seq_upper.count("C")
        valid_count = gc_count + seq_upper.count("A") + seq_upper.count("T")

        if valid_count == 0:
            return 0.0
        return round((gc_count / valid_count) * 100.0, 2)

    @classmethod
    def validate_record(
        cls, chromosome_id: str, sequence: str, min_length: int = 1
    ) -> Tuple[bool, str]:
        """Validate that a chromosome record meets basic downstream requirements."""
        if not chromosome_id or not chromosome_id.strip():
            return False, "Chromosome ID cannot be empty."

        if len(sequence) < min_length:
            return False, f"Sequence length ({len(sequence)}) is below minimum ({min_length})."

        return True, ""

