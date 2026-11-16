"""
BioPython-powered FASTA reader and genomic parsing pipeline for GeneWeaver.

Uses BioPython's Bio.SeqIO engine to stream and parse raw or gzipped
human genome FASTA files, cleaning sequences and dispatching chunks as NumPy arrays.
"""

from dataclasses import dataclass
import gzip
import os
from typing import Generator, Iterator, List, Optional, Tuple, Union

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from genomic_parser.metadata_extractor import (
    ChromosomeMetadata,
    GenomeMetadata,
    MetadataExtractor,
)
from genomic_parser.sequence_cleaner import SequenceCleaner
from genomic_parser.sequence_chunker import GenomicChunk, SequenceChunker
from genomic_parser.sequence_validator import SequenceValidator


@dataclass
class ParsedChromosome:
    """A fully parsed chromosome record with metadata."""
    record_id: str
    description: str
    sequence: str
    metadata: ChromosomeMetadata


class FastaReader:
    """
    Reads and streams genomic FASTA sequences using BioPython.

    Supports both uncompressed (.fasta, .fa) and gzip-compressed (.fasta.gz, .fa.gz) files.
    """

    def __init__(self, file_path: str, clean_sequences: bool = True):
        """
        Initialize the FASTA reader.

        Args:
            file_path: Path to the FASTA file.
            clean_sequences: Whether to automatically uppercase and sanitize sequences.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FASTA file not found: {file_path}")

        self.file_path = os.path.abspath(file_path)
        self.clean_sequences = clean_sequences
        self.is_gzipped = file_path.endswith(".gz")

    def _open_handle(self):
        """Open the appropriate file handle depending on compression."""
        if self.is_gzipped:
            return gzip.open(self.file_path, "rt", encoding="utf-8")
        return open(self.file_path, "r", encoding="utf-8")

    def stream_records(self) -> Iterator[SeqRecord]:
        """
        Stream raw BioPython SeqRecord objects one by one without loading entire file.

        Yields:
            Bio.SeqRecord.SeqRecord objects.
        """
        with self._open_handle() as handle:
            for record in SeqIO.parse(handle, "fasta"):
                yield record

    def stream_chromosomes(self) -> Iterator[ParsedChromosome]:
        """
        Stream cleaned chromosomes with metadata.

        Yields:
            ParsedChromosome instances.
        """
        for record in self.stream_records():
            raw_seq = str(record.seq)
            clean_seq = (
                SequenceCleaner.clean(raw_seq) if self.clean_sequences else raw_seq
            )

            metadata = MetadataExtractor.extract_chromosome_metadata(
                chromosome_id=record.id,
                sequence=clean_seq,
                description=record.description,
            )

            yield ParsedChromosome(
                record_id=record.id,
                description=record.description,
                sequence=clean_seq,
                metadata=metadata,
            )

    def read_all(self) -> List[ParsedChromosome]:
        """Load and parse all chromosomes into memory."""
        return list(self.stream_chromosomes())

    def get_genome_metadata(self) -> GenomeMetadata:
        """Parse all headers and sequence statistics to produce GenomeMetadata."""
        meta_list = [c.metadata for c in self.stream_chromosomes()]
        return MetadataExtractor.summarize_genome(meta_list)


class GenomePipeline:
    """
    End-to-end data pipeline: Uses BioPython to parse a human genome file
    and chunks each chromosome into manageable NumPy arrays for downstream alignment.
    """

    def __init__(
        self,
        fasta_path: str,
        chunk_size: int = 10_000,
        overlap: int = 100,
        clean_sequences: bool = True,
    ):
        """
        Initialize the genomic data pipeline.

        Args:
            fasta_path: Path to FASTA file (plain or .gz).
            chunk_size: Target length for each chunk array in nucleotides.
            overlap: Overlap length between adjacent chunks for boundary coverage.
            clean_sequences: Whether to normalize characters to uppercase ACGTN.
        """
        self.reader = FastaReader(fasta_path, clean_sequences=clean_sequences)
        self.chunker = SequenceChunker(chunk_size=chunk_size, overlap=overlap)

    def stream_chunks(self) -> Iterator[GenomicChunk]:
        """
        Stream all genomic chunks across all chromosomes.

        Yields:
            GenomicChunk objects containing NumPy arrays (dtype uint8) and coordinates.
        """
        for chrom in self.reader.stream_chromosomes():
            yield from self.chunker.iter_chunks(
                chromosome_id=chrom.record_id,
                sequence=chrom.sequence,
            )

    def process_all_chunks(self) -> List[GenomicChunk]:
        """Process and collect all chunks across the entire genome in memory."""
        return list(self.stream_chunks())

    def get_summary(self) -> GenomeMetadata:
        """Get summary metadata for the parsed genome."""
        return self.reader.get_genome_metadata()

