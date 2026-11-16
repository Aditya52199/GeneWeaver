"""
Metadata extraction for genomic FASTA sequences.

Analyzes contigs/chromosomes to extract summary statistics, composition,
and coordinate bounds for alignment scheduling.
"""

from dataclasses import dataclass
from typing import Dict, List
from genomic_parser.sequence_validator import SequenceValidator


@dataclass
class ChromosomeMetadata:
    """Summary statistics for an individual chromosome/contig."""
    name: str
    description: str
    length: int
    gc_content: float
    composition: Dict[str, int]
    ambiguous_percentage: float


@dataclass
class GenomeMetadata:
    """Summary statistics for an entire parsed genome dataset."""
    total_length: int
    chromosome_count: int
    chromosomes: List[ChromosomeMetadata]
    average_gc_content: float
    total_ambiguous_bases: int


class MetadataExtractor:
    """Extracts summary and positional metadata from parsed chromosomes."""

    @staticmethod
    def extract_chromosome_metadata(
        chromosome_id: str,
        sequence: str,
        description: str = "",
    ) -> ChromosomeMetadata:
        """Analyze a single sequence and produce its ChromosomeMetadata."""
        length = len(sequence)
        composition = SequenceValidator.calculate_composition(sequence)
        gc_content = SequenceValidator.calculate_gc_content(sequence)

        n_count = composition.get("N", 0) + composition.get("other", 0)
        ambiguous_pct = round((n_count / length * 100.0), 2) if length > 0 else 0.0

        return ChromosomeMetadata(
            name=chromosome_id,
            description=description,
            length=length,
            gc_content=gc_content,
            composition=composition,
            ambiguous_percentage=ambiguous_pct,
        )

    @classmethod
    def summarize_genome(
        cls, chromosome_metadata_list: List[ChromosomeMetadata]
    ) -> GenomeMetadata:
        """Aggregate individual chromosome metrics into overall GenomeMetadata."""
        total_len = sum(c.length for c in chromosome_metadata_list)
        total_ambiguous = sum(
            c.composition.get("N", 0) + c.composition.get("other", 0)
            for c in chromosome_metadata_list
        )

        total_gc_bases = sum(
            c.composition.get("G", 0) + c.composition.get("C", 0)
            for c in chromosome_metadata_list
        )
        valid_bases = sum(
            c.composition.get("A", 0)
            + c.composition.get("C", 0)
            + c.composition.get("G", 0)
            + c.composition.get("T", 0)
            for c in chromosome_metadata_list
        )
        avg_gc = round((total_gc_bases / valid_bases * 100.0), 2) if valid_bases > 0 else 0.0

        return GenomeMetadata(
            total_length=total_len,
            chromosome_count=len(chromosome_metadata_list),
            chromosomes=chromosome_metadata_list,
            average_gc_content=avg_gc,
            total_ambiguous_bases=total_ambiguous,
        )

