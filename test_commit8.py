
import unittest
from unittest.mock import patch

from geneweaver import GenomicChunk, scan_chunk
from validation import (
    validate_guide,
    compare_candidates,
    validate_chunk
)


GUIDE = b"ACGTACGTACGTACGTACGT"


def make_test_chunk():
    sequence = (
        b"A" * 5
        + GUIDE
        + b"AGG"
        + b"A" * 20
    )

    return GenomicChunk(
        chromosome="test_chromosome",
        start_position=100,
        sequence=sequence,
        valid_starts=len(sequence) - 22,
        chunk_index=0
    )


class TestCommit8(unittest.TestCase):

    def test_valid_guide(self):
        result = validate_guide(
            "ACGTACGTACGTACGTACGT"
        )

        self.assertEqual(result, GUIDE)

    def test_invalid_guide_length(self):
        with self.assertRaises(ValueError):
            validate_guide("ACGT")

    def test_invalid_guide_character(self):
        with self.assertRaises(ValueError):
            validate_guide("ACGTACGTACGTACGTACGN")

    def test_identical_candidate_results(self):
        chunk = make_test_chunk()

        candidates, stats = scan_chunk(
            chunk,
            GUIDE,
            3
        )

        comparison = compare_candidates(
            candidates,
            candidates.copy()
        )

        self.assertTrue(comparison["matches"])
        self.assertEqual(
            comparison["missing_count"],
            0
        )
        self.assertEqual(
            comparison["extra_count"],
            0
        )

    def test_missing_candidate_detection(self):
        chunk = make_test_chunk()

        candidates, stats = scan_chunk(
            chunk,
            GUIDE,
            3
        )

        self.assertGreater(len(candidates), 0)

        comparison = compare_candidates(
            candidates,
            candidates[1:]
        )

        self.assertFalse(comparison["matches"])
        self.assertEqual(
            comparison["missing_count"],
            1
        )

    def test_mismatch_position_difference(self):
        chunk = make_test_chunk()

        candidates, stats = scan_chunk(
            chunk,
            GUIDE,
            3
        )

        self.assertGreater(len(candidates), 0)

        modified_candidates = [
            dict(candidate)
            for candidate in candidates
        ]

        modified_candidates[0]["mismatch_positions"] = "1,2"

        comparison = compare_candidates(
            candidates,
            modified_candidates
        )

        self.assertFalse(comparison["matches"])

    def test_cpu_fallback_validation(self):
        chunk = make_test_chunk()

        expected_candidates, expected_stats = scan_chunk(
            chunk,
            GUIDE,
            3
        )

        fake_result = {
            "backend": "CPU",
            "candidates": expected_candidates,
            "statistics": expected_stats
        }

        with patch(
            "validation.scan_chunk_accelerated",
            return_value=fake_result
        ):
            result = validate_chunk(
                chunk,
                GUIDE,
                3,
                5
            )

        self.assertTrue(result["validation_passed"])
        self.assertEqual(result["backend"], "CPU")

    def test_validation_detects_pam_count_difference(self):
        chunk = make_test_chunk()

        expected_candidates, expected_stats = scan_chunk(
            chunk,
            GUIDE,
            3
        )

        fake_stats = dict(expected_stats)
        fake_stats["pam_sites"] += 1

        fake_result = {
            "backend": "CPU",
            "candidates": expected_candidates,
            "statistics": fake_stats
        }

        with patch(
            "validation.scan_chunk_accelerated",
            return_value=fake_result
        ):
            result = validate_chunk(
                chunk,
                GUIDE,
                3,
                5
            )

        self.assertFalse(result["validation_passed"])
        self.assertFalse(
            result["comparison"]["pam_count_matches"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
