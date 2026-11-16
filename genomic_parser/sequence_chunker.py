"""
Genomic sequence chunking engine for GeneWeaver.

Chunks full chromosome sequences into manageable contiguous NumPy arrays
with configurable overlap/padding for boundary-safe CRISPR PAM and off-target alignment.
"""

from dataclasses import dataclass
from typing import Generator, Iterator, List, Optional
import numpy as np

from genomic_parser.sequence_validator import SequenceValidator


@dataclass(frozen=True)
class GenomicChunk:
    """
    Represents a discrete chunk of genomic sequence ready for CPU or GPU processing.

    Attributes:
        chromosome: Name or accession of the chromosome/contig (e.g. 'chr1').
        chunk_id: 0-indexed chunk identifier within the chromosome.
        start_pos: 0-based start coordinate in the parent chromosome.
        end_pos: 0-based exclusive end coordinate in the parent chromosome.
        array: NumPy array of shape (length,) with dtype uint8 (ASCII byte representation).
        gc_content: GC percentage of this chunk.
        ambiguous_count: Number of 'N' or non-canonical bases in this chunk.
    """
    chromosome: str
    chunk_id: int
    start_pos: int
    end_pos: int
    array: np.ndarray
    gc_content: float
    ambiguous_count: int

    @property
    def length(self) -> int:
        """Length of this chunk in base pairs."""
        return len(self.array)

    def as_str(self) -> str:
        """Decode the underlying byte array back into a Python DNA string."""
        return bytes(self.array).decode("ascii")

    def __repr__(self) -> str:
        return (
            f"GenomicChunk(chr={self.chromosome}, id={self.chunk_id}, "
            f"range=[{self.start_pos}:{self.end_pos}], len={self.length}, "
            f"GC={self.gc_content:.1f}%)"
        )


class SequenceChunker:
    """
    Slices long chromosome sequences into overlapping, manageable NumPy arrays.

    Overlapping prevents false-negative alignment cutoffs at chunk boundaries
    (crucial for CRISPR 20bp guide RNA + 3bp PAM detection).
    """

    def __init__(
        self,
        chunk_size: int = 10_000,
        overlap: int = 100,
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target size (in nucleotides) for each chunk (default: 10,000).
            overlap: Overlap length (in nucleotides) between consecutive chunks (default: 100).
                     Must be strictly less than chunk_size.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap cannot be negative, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.step = chunk_size - overlap

    def chunk_sequence(
        self,
        chromosome_id: str,
        sequence: str,
    ) -> List[GenomicChunk]:
        """
        Chunk a sequence in-memory and return a list of GenomicChunk objects.

        Args:
            chromosome_id: Name of the chromosome/contig.
            sequence: Raw or cleaned DNA sequence string.

        Returns:
            List of GenomicChunk objects.
        """
        return list(self.iter_chunks(chromosome_id, sequence))

    def iter_chunks(
        self,
        chromosome_id: str,
        sequence: str,
    ) -> Iterator[GenomicChunk]:
        """
        Generator yielding GenomicChunk objects one by one without creating all arrays upfront.

        Args:
            chromosome_id: Name of the chromosome/contig.
            sequence: DNA sequence string.

        Yields:
            GenomicChunk instances.
        """
        total_len = len(sequence)
        if total_len == 0:
            return

        # If sequence is smaller than chunk size, return single chunk
        if total_len <= self.chunk_size:
            yield self._create_chunk(chromosome_id, 0, 0, total_len, sequence)
            return

        chunk_id = 0
        start = 0

        while start < total_len:
            end = min(start + self.chunk_size, total_len)
            sub_seq = sequence[start:end]

            yield self._create_chunk(chromosome_id, chunk_id, start, end, sub_seq)

            chunk_id += 1
            if end >= total_len:
                break
            start += self.step

    @staticmethod
    def _create_chunk(
        chromosome_id: str,
        chunk_id: int,
        start_pos: int,
        end_pos: int,
        sub_sequence: str,
    ) -> GenomicChunk:
        """Helper to construct a GenomicChunk with NumPy array and metrics."""
        # Convert ASCII string to 1D uint8 NumPy array
        byte_arr = np.frombuffer(sub_sequence.encode("ascii"), dtype=np.uint8).copy()

        gc_content = SequenceValidator.calculate_gc_content(sub_sequence)
        ambiguous_count = sub_sequence.upper().count("N")

        return GenomicChunk(
            chromosome=chromosome_id,
            chunk_id=chunk_id,
            start_pos=start_pos,
            end_pos=end_pos,
            array=byte_arr,
            gc_content=gc_content,
            ambiguous_count=ambiguous_count,
        )

