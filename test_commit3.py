
import unittest

from geneweaver import (
    GenomicChunk,
    scan_chunk,
    reverse_complement
)


GUIDE = b"ACGTACGTACGTACGTACGT"


class TestCommit3(unittest.TestCase):

    def test_forward_strand_match(self):
        sequence = GUIDE + b"AGG"

        chunk = GenomicChunk(
            chromosome="test",
            start_position=0,
            sequence=sequence,
            valid_starts=1,
            chunk_index=0
        )

        results, stats = scan_chunk(chunk, GUIDE, 0)

        self.assertTrue(
            any(result["strand"] == "+" for result in results)
        )

        self.assertEqual(len(results), 1)

    def test_reverse_strand_match(self):
        oriented_sequence = GUIDE + b"AGG"
        reference_sequence = reverse_complement(oriented_sequence)

        chunk = GenomicChunk(
            chromosome="test",
            start_position=100,
            sequence=reference_sequence,
            valid_starts=1,
            chunk_index=0
        )

        results, stats = scan_chunk(chunk, GUIDE, 0)

        self.assertTrue(
            any(result["strand"] == "-" for result in results)
        )

    def test_mismatch_limit(self):
        target = bytearray(GUIDE)
        target[0] = ord("T") if target[0] != ord("T") else ord("A")

        sequence = bytes(target) + b"AGG"

        chunk = GenomicChunk(
            chromosome="test",
            start_position=0,
            sequence=sequence,
            valid_starts=1,
            chunk_index=0
        )

        results, stats = scan_chunk(chunk, GUIDE, 0)

        self.assertEqual(len(results), 0)

    def test_chunk_boundary_match(self):
        prefix = b"A" * 90
        sequence = prefix + GUIDE + b"AGG" + b"A" * 30

        # The chunk owns starts 0 through 99.
        # The 23-base match starts at position 90 and crosses
        # the boundary at position 100.
        chunk = GenomicChunk(
            chromosome="test",
            start_position=0,
            sequence=sequence[:123],
            valid_starts=100,
            chunk_index=0
        )

        results, stats = scan_chunk(chunk, GUIDE, 0)

        matching_positions = [
            result["start_position"]
            for result in results
            if result["strand"] == "+"
        ]

        self.assertIn(90, matching_positions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
