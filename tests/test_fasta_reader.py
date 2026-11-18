"""
Unit tests for BioPython FASTA reader and Genomic Pipeline.
"""

import os
import unittest
import numpy as np

from genomic_parser.fasta_reader import FastaReader, GenomePipeline
from genomic_parser.sequence_chunker import SequenceChunker
from genomic_parser.sequence_cleaner import SequenceCleaner
from genomic_parser.sequence_validator import SequenceValidator


class TestGenomicPipeline(unittest.TestCase):
    """Test suite for BioPython ingestion, cleaning, and NumPy chunking."""

    @classmethod
    def setUpClass(cls):
        cls.fasta_path = "data/mock_human_genome.fasta"
        cls.gz_fasta_path = "data/mock_human_genome.fasta.gz"

    def test_fasta_reader_stream_records(self):
        """Verify BioPython streams all chromosome records."""
        reader = FastaReader(self.fasta_path)
        records = list(reader.stream_records())

        self.assertEqual(len(records), 4)
        chr_ids = [r.id for r in records]
        self.assertEqual(chr_ids, ["chr1", "chr2", "chr21", "chrX"])

    def test_fasta_reader_gzipped(self):
        """Verify BioPython can read compressed .fasta.gz files."""
        reader = FastaReader(self.gz_fasta_path)
        chroms = list(reader.stream_chromosomes())

        self.assertEqual(len(chroms), 4)
        self.assertEqual(chroms[0].record_id, "chr1")
        self.assertEqual(chroms[0].metadata.length, 60_000)

    def test_sequence_cleaner(self):
        """Verify sequence normalization: uppercase conversion, whitespace stripping."""
        raw = "acgt  \n  TGCA n\n xyz"
        cleaned = SequenceCleaner.clean(raw, convert_to_uppercase=True, mask_ambiguous_as_n=True)
        # xyz should be converted to NNN
        self.assertEqual(cleaned, "ACGTTGCANNNN")

    def test_sequence_validator(self):
        """Verify GC-content calculation and nucleotide composition."""
        seq = "GGCC" + "AATT"  # 4 GC, 4 AT = 50% GC
        gc = SequenceValidator.calculate_gc_content(seq)
        self.assertEqual(gc, 50.0)

        comp = SequenceValidator.calculate_composition(seq)
        self.assertEqual(comp["G"], 2)
        self.assertEqual(comp["C"], 2)
        self.assertEqual(comp["A"], 2)
        self.assertEqual(comp["T"], 2)

    def test_sequence_chunker_basic(self):
        """Verify chunker produces expected number of chunks with proper overlap."""
        seq = "A" * 1000
        chunker = SequenceChunker(chunk_size=300, overlap=50)
        chunks = chunker.chunk_sequence("chr_test", seq)

        # Step size = 300 - 50 = 250
        # Chunk 0: [0:300]
        # Chunk 1: [250:550]
        # Chunk 2: [500:800]
        # Chunk 3: [750:1000]
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0].start_pos, 0)
        self.assertEqual(chunks[0].end_pos, 300)
        self.assertEqual(chunks[1].start_pos, 250)
        self.assertEqual(chunks[1].end_pos, 550)
        self.assertEqual(chunks[3].end_pos, 1000)

    def test_chunk_numpy_array(self):
        """Verify chunk contains a valid NumPy uint8 array matching the ASCII DNA."""
        dna = "ATGC" * 10
        chunker = SequenceChunker(chunk_size=20, overlap=5)
        chunks = chunker.chunk_sequence("chr1", dna)

        first_chunk = chunks[0]
        self.assertIsInstance(first_chunk.array, np.ndarray)
        self.assertEqual(first_chunk.array.dtype, np.uint8)
        self.assertEqual(first_chunk.as_str(), dna[:20])

    def test_genome_pipeline_end_to_end(self):
        """Verify complete pipeline parsing mock genome and yielding chunks."""
        pipeline = GenomePipeline(
            fasta_path=self.fasta_path,
            chunk_size=10_000,
            overlap=100,
        )

        chunks = pipeline.process_all_chunks()
        summary = pipeline.get_summary()

        self.assertEqual(summary.chromosome_count, 4)
        self.assertEqual(summary.total_length, 160_000)
        self.assertGreater(len(chunks), 0)

        # Verify all chunks have valid coordinates and non-empty NumPy arrays
        for chunk in chunks:
            self.assertGreater(chunk.length, 0)
            self.assertEqual(len(chunk.array), chunk.end_pos - chunk.start_pos)
            self.assertIn(chunk.chromosome, ["chr1", "chr2", "chr21", "chrX"])


if __name__ == "__main__":
    unittest.main()

