
import unittest
from benchmark_cpu import validate_guide


class TestCommit4(unittest.TestCase):

    def test_valid_guide(self):
        guide = "ACGTACGTACGTACGTACGT"

        result = validate_guide(guide)

        self.assertEqual(result, guide.encode("ascii"))

    def test_lowercase_guide(self):
        guide = "acgtacgtacgtacgtacgt"

        result = validate_guide(guide)

        self.assertEqual(result, guide.upper().encode("ascii"))

    def test_invalid_guide_length(self):
        with self.assertRaises(ValueError):
            validate_guide("ACGT")

    def test_invalid_guide_character(self):
        with self.assertRaises(ValueError):
            validate_guide("ACGTACGTACGTACGTACGN")


if __name__ == "__main__":
    unittest.main(verbosity=2)
