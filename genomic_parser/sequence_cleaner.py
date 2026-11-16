"""
Genomic sequence cleaner and normalization utilities.

Cleans raw genomic strings from FASTA files:
- Normalizes soft-masked repeats to uppercase (a,c,g,t -> A,C,G,T)
- Resolves or flags ambiguous/IUPAC nucleotide characters (N, R, Y, etc.)
- Strips invalid characters, whitespaces, and newlines
"""

import re
from typing import Set, Tuple

# Standard canonical DNA bases
CANONICAL_BASES: Set[str] = {"A", "C", "G", "T"}
VALID_IUPAC_BASES: Set[str] = {
    "A", "C", "G", "T", "N", "R", "Y", "S", "W", "K", "M", "B", "D", "H", "V"
}

_INVALID_CHARS_REGEX = re.compile(r"[^ACGTN]")


class SequenceCleaner:
    """Utilities for sanitizing and standardizing raw genomic sequences."""

    @staticmethod
    def clean(
        sequence: str,
        convert_to_uppercase: bool = True,
        mask_ambiguous_as_n: bool = True,
    ) -> str:
        """
        Clean and normalize raw genomic sequence text.

        Args:
            sequence: Raw sequence string (can include lowercase, spaces, newlines).
            convert_to_uppercase: Convert all characters to uppercase.
            mask_ambiguous_as_n: Replace any non-ACGT character with 'N'.

        Returns:
            Sanitized DNA string.
        """
        if not sequence:
            return ""

        # Remove whitespace and newlines
        cleaned = "".join(sequence.split())

        if convert_to_uppercase:
            cleaned = cleaned.upper()

        if mask_ambiguous_as_n:
            # Replace non-ACGT characters with N
            cleaned = _INVALID_CHARS_REGEX.sub("N", cleaned)

        return cleaned

    @staticmethod
    def validate_dna(sequence: str, allow_n: bool = True) -> Tuple[bool, str]:
        """
        Validate whether a sequence contains valid DNA bases.

        Args:
            sequence: Sequence string to validate.
            allow_n: Whether 'N' (unsequenced/ambiguous base) is considered valid.

        Returns:
            Tuple of (is_valid, error_message).
        """
        allowed = CANONICAL_BASES.union({"N"}) if allow_n else CANONICAL_BASES
        seq_upper = sequence.upper()
        invalid_chars = set(seq_upper) - allowed
        if invalid_chars:
            return False, f"Sequence contains invalid bases: {sorted(list(invalid_chars))}"
        return True, ""

