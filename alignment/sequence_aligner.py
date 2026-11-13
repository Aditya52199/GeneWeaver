"""
Pure Python CPU Baseline Sequence Aligner for GeneWeaver.

Implements standard dynamic programming DNA sequence alignment algorithms:
- Needleman-Wunsch (Global Alignment)
- Smith-Waterman (Local Alignment)
- Semi-Global Alignment (End-free gaps for CRISPR guide-RNA matching)

These pure Python implementations establish baseline accuracy and performance
benchmarks (execution time, MCUPS) to evaluate subsequent GPU-accelerated
and distributed implementations.
"""

from dataclasses import dataclass
from enum import Enum
import time
from typing import List, Optional, Tuple


class AlignmentType(str, Enum):
    """Supported alignment algorithm types."""
    GLOBAL = "global"          # Needleman-Wunsch
    LOCAL = "local"            # Smith-Waterman
    SEMI_GLOBAL = "semi_global"  # Free end gaps (guide-to-target)


# Traceback direction bitmasks
STOP = 0
DIAG = 1   # Match / Mismatch
UP = 2     # Deletion (gap in seq2)
LEFT = 3   # Insertion (gap in seq1)


@dataclass
class AlignmentResult:
    """Stores the complete output and performance metadata of an alignment."""
    seq1_aligned: str
    seq2_aligned: str
    score: int
    alignment_type: str
    execution_time_seconds: float
    seq1_length: int
    seq2_length: int
    cells_computed: int
    mcups: float  # Mega Cell Updates Per Second
    start_pos: Tuple[int, int]
    end_pos: Tuple[int, int]
    matches: int
    mismatches: int
    gaps: int
    identity_percent: float
    cigar: str

    @property
    def execution_time_ms(self) -> float:
        """Execution time in milliseconds."""
        return self.execution_time_seconds * 1000.0

    def formatted_alignment(self, line_length: int = 60) -> str:
        """Format the alignment into a standard human-readable 3-line visualization."""
        midline_chars = []
        for c1, c2 in zip(self.seq1_aligned, self.seq2_aligned):
            if c1 == c2 and c1 != "-":
                midline_chars.append("|")
            elif c1 == "-" or c2 == "-":
                midline_chars.append(" ")
            else:
                midline_chars.append(".")
        midline = "".join(midline_chars)

        lines = []
        total_len = len(self.seq1_aligned)
        for i in range(0, total_len, line_length):
            chunk1 = self.seq1_aligned[i:i + line_length]
            chunk_mid = midline[i:i + line_length]
            chunk2 = self.seq2_aligned[i:i + line_length]
            lines.append(f"Seq1: {chunk1}")
            lines.append(f"      {chunk_mid}")
            lines.append(f"Seq2: {chunk2}\n")
        return "\n".join(lines).rstrip()


class SequenceAligner:
    """
    Standard DNA Sequence Aligner implemented in pure Python.

    Provides exact dynamic programming alignment without external dependencies.
    Serves as the golden standard for correctness and CPU baseline timing.
    """

    def __init__(
        self,
        match_score: int = 2,
        mismatch_penalty: int = -1,
        gap_penalty: int = -2,
    ):
        """
        Initialize the aligner with scoring parameters.

        Args:
            match_score: Score awarded for matching nucleotides (default: +2).
            mismatch_penalty: Penalty for mismatched nucleotides (default: -1).
            gap_penalty: Linear penalty for introducing an indel/gap (default: -2).
        """
        if match_score <= 0:
            raise ValueError("match_score must be positive.")
        if mismatch_penalty > 0:
            raise ValueError("mismatch_penalty must be non-positive.")
        if gap_penalty > 0:
            raise ValueError("gap_penalty must be non-positive.")

        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty

    def _score(self, char1: str, char2: str) -> int:
        """Compute substitution score between two characters."""
        return self.match_score if char1 == char2 else self.mismatch_penalty

    def align(
        self,
        seq1: str,
        seq2: str,
        mode: AlignmentType = AlignmentType.GLOBAL,
    ) -> AlignmentResult:
        """
        Align two DNA sequences using the specified alignment mode.

        Args:
            seq1: First sequence (e.g. query / guide RNA).
            seq2: Second sequence (e.g. target DNA).
            mode: AlignmentType (GLOBAL, LOCAL, or SEMI_GLOBAL).

        Returns:
            AlignmentResult with aligned sequences, score, and execution metrics.
        """
        seq1 = seq1.strip().upper()
        seq2 = seq2.strip().upper()

        start_time = time.perf_counter()

        if mode == AlignmentType.GLOBAL:
            aligned1, aligned2, score, start_pos, end_pos, cells = (
                self._needleman_wunsch(seq1, seq2)
            )
        elif mode == AlignmentType.LOCAL:
            aligned1, aligned2, score, start_pos, end_pos, cells = (
                self._smith_waterman(seq1, seq2)
            )
        elif mode == AlignmentType.SEMI_GLOBAL:
            aligned1, aligned2, score, start_pos, end_pos, cells = (
                self._semi_global(seq1, seq2)
            )
        else:
            raise ValueError(f"Unsupported alignment mode: {mode}")

        elapsed_time = time.perf_counter() - start_time

        # Calculate alignment metrics
        matches = 0
        mismatches = 0
        gaps = 0
        for c1, c2 in zip(aligned1, aligned2):
            if c1 == "-" or c2 == "-":
                gaps += 1
            elif c1 == c2:
                matches += 1
            else:
                mismatches += 1

        total_aligned_len = len(aligned1)
        identity = (matches / total_aligned_len * 100.0) if total_aligned_len > 0 else 0.0

        # MCUPS: (cells computed / elapsed seconds) / 1,000,000
        mcups = (cells / (elapsed_time * 1e6)) if elapsed_time > 0 else 0.0

        cigar = self._generate_cigar(aligned1, aligned2)

        return AlignmentResult(
            seq1_aligned=aligned1,
            seq2_aligned=aligned2,
            score=score,
            alignment_type=mode.value if isinstance(mode, AlignmentType) else str(mode),
            execution_time_seconds=elapsed_time,
            seq1_length=len(seq1),
            seq2_length=len(seq2),
            cells_computed=cells,
            mcups=mcups,
            start_pos=start_pos,
            end_pos=end_pos,
            matches=matches,
            mismatches=mismatches,
            gaps=gaps,
            identity_percent=round(identity, 2),
            cigar=cigar,
        )

    def _needleman_wunsch(
        self, seq1: str, seq2: str
    ) -> Tuple[str, str, int, Tuple[int, int], Tuple[int, int], int]:
        """Global sequence alignment via standard Needleman-Wunsch algorithm."""
        m = len(seq1)
        n = len(seq2)
        cells_computed = (m + 1) * (n + 1)

        # Initialize DP matrix and traceback matrix
        score_matrix = [[0] * (n + 1) for _ in range(m + 1)]
        traceback = [[STOP] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            score_matrix[i][0] = i * self.gap_penalty
            traceback[i][0] = UP

        for j in range(1, n + 1):
            score_matrix[0][j] = j * self.gap_penalty
            traceback[0][j] = LEFT

        # Fill DP matrix
        for i in range(1, m + 1):
            char1 = seq1[i - 1]
            row_prev = score_matrix[i - 1]
            row_curr = score_matrix[i]
            tb_row = traceback[i]

            for j in range(1, n + 1):
                char2 = seq2[j - 1]
                diag_score = row_prev[j - 1] + self._score(char1, char2)
                up_score = row_prev[j] + self.gap_penalty
                left_score = row_curr[j - 1] + self.gap_penalty

                # Priority: DIAG >= UP >= LEFT
                if diag_score >= up_score and diag_score >= left_score:
                    row_curr[j] = diag_score
                    tb_row[j] = DIAG
                elif up_score >= left_score:
                    row_curr[j] = up_score
                    tb_row[j] = UP
                else:
                    row_curr[j] = left_score
                    tb_row[j] = LEFT

        # Traceback from (m, n) to (0, 0)
        aligned1 = []
        aligned2 = []
        curr_i = m
        curr_j = n

        while curr_i > 0 or curr_j > 0:
            direction = traceback[curr_i][curr_j]
            if direction == DIAG:
                aligned1.append(seq1[curr_i - 1])
                aligned2.append(seq2[curr_j - 1])
                curr_i -= 1
                curr_j -= 1
            elif direction == UP:
                aligned1.append(seq1[curr_i - 1])
                aligned2.append("-")
                curr_i -= 1
            elif direction == LEFT:
                aligned1.append("-")
                aligned2.append(seq2[curr_j - 1])
                curr_j -= 1
            else:
                # Top-left corner reached or fallback
                if curr_i > 0:
                    aligned1.append(seq1[curr_i - 1])
                    aligned2.append("-")
                    curr_i -= 1
                elif curr_j > 0:
                    aligned1.append("-")
                    aligned2.append(seq2[curr_j - 1])
                    curr_j -= 1

        aligned1.reverse()
        aligned2.reverse()

        final_score = score_matrix[m][n]
        return "".join(aligned1), "".join(aligned2), final_score, (0, 0), (m, n), cells_computed

    def _smith_waterman(
        self, seq1: str, seq2: str
    ) -> Tuple[str, str, int, Tuple[int, int], Tuple[int, int], int]:
        """Local sequence alignment via standard Smith-Waterman algorithm."""
        m = len(seq1)
        n = len(seq2)
        cells_computed = (m + 1) * (n + 1)

        score_matrix = [[0] * (n + 1) for _ in range(m + 1)]
        traceback = [[STOP] * (n + 1) for _ in range(m + 1)]

        max_score = 0
        max_i, max_j = 0, 0

        # Fill DP matrix with non-negative lower bound
        for i in range(1, m + 1):
            char1 = seq1[i - 1]
            row_prev = score_matrix[i - 1]
            row_curr = score_matrix[i]
            tb_row = traceback[i]

            for j in range(1, n + 1):
                char2 = seq2[j - 1]
                diag_score = row_prev[j - 1] + self._score(char1, char2)
                up_score = row_prev[j] + self.gap_penalty
                left_score = row_curr[j - 1] + self.gap_penalty

                best_score = 0
                best_direction = STOP

                if diag_score > best_score:
                    best_score = diag_score
                    best_direction = DIAG
                if up_score > best_score:
                    best_score = up_score
                    best_direction = UP
                if left_score > best_score:
                    best_score = left_score
                    best_direction = LEFT

                row_curr[j] = best_score
                tb_row[j] = best_direction

                if best_score > max_score:
                    max_score = best_score
                    max_i = i
                    max_j = j

        # Traceback from highest score down to score <= 0 or STOP
        aligned1 = []
        aligned2 = []
        curr_i = max_i
        curr_j = max_j

        while curr_i > 0 and curr_j > 0 and score_matrix[curr_i][curr_j] > 0:
            direction = traceback[curr_i][curr_j]
            if direction == DIAG:
                aligned1.append(seq1[curr_i - 1])
                aligned2.append(seq2[curr_j - 1])
                curr_i -= 1
                curr_j -= 1
            elif direction == UP:
                aligned1.append(seq1[curr_i - 1])
                aligned2.append("-")
                curr_i -= 1
            elif direction == LEFT:
                aligned1.append("-")
                aligned2.append(seq2[curr_j - 1])
                curr_j -= 1
            else:
                break

        aligned1.reverse()
        aligned2.reverse()

        start_pos = (curr_i, curr_j)
        end_pos = (max_i, max_j)

        return "".join(aligned1), "".join(aligned2), max_score, start_pos, end_pos, cells_computed

    def _semi_global(
        self, seq1: str, seq2: str
    ) -> Tuple[str, str, int, Tuple[int, int], Tuple[int, int], int]:
        """
        Semi-global alignment: Free start and end gaps in seq2 (target DNA).
        Ideal for finding where a short guide RNA (seq1) aligns inside target seq2.
        """
        m = len(seq1)
        n = len(seq2)
        cells_computed = (m + 1) * (n + 1)

        score_matrix = [[0] * (n + 1) for _ in range(m + 1)]
        traceback = [[STOP] * (n + 1) for _ in range(m + 1)]

        # Free start gaps in seq2 (top row stays 0)
        for i in range(1, m + 1):
            score_matrix[i][0] = i * self.gap_penalty
            traceback[i][0] = UP

        for i in range(1, m + 1):
            char1 = seq1[i - 1]
            row_prev = score_matrix[i - 1]
            row_curr = score_matrix[i]
            tb_row = traceback[i]

            for j in range(1, n + 1):
                char2 = seq2[j - 1]
                diag_score = row_prev[j - 1] + self._score(char1, char2)
                up_score = row_prev[j] + self.gap_penalty
                left_score = row_curr[j - 1] + self.gap_penalty

                if diag_score >= up_score and diag_score >= left_score:
                    row_curr[j] = diag_score
                    tb_row[j] = DIAG
                elif up_score >= left_score:
                    row_curr[j] = up_score
                    tb_row[j] = UP
                else:
                    row_curr[j] = left_score
                    tb_row[j] = LEFT

        # Find maximum in bottom row (seq1 fully aligned, free trailing gaps in seq2)
        best_j = n
        best_score = score_matrix[m][n]
        for j in range(1, n + 1):
            if score_matrix[m][j] > best_score:
                best_score = score_matrix[m][j]
                best_j = j

        aligned1 = []
        aligned2 = []
        curr_i = m
        curr_j = best_j

        while curr_i > 0 and curr_j > 0:
            direction = traceback[curr_i][curr_j]
            if direction == DIAG:
                aligned1.append(seq1[curr_i - 1])
                aligned2.append(seq2[curr_j - 1])
                curr_i -= 1
                curr_j -= 1
            elif direction == UP:
                aligned1.append(seq1[curr_i - 1])
                aligned2.append("-")
                curr_i -= 1
            elif direction == LEFT:
                aligned1.append("-")
                aligned2.append(seq2[curr_j - 1])
                curr_j -= 1
            else:
                break

        # Handle remaining prefix if any
        while curr_i > 0:
            aligned1.append(seq1[curr_i - 1])
            aligned2.append("-")
            curr_i -= 1

        aligned1.reverse()
        aligned2.reverse()

        return "".join(aligned1), "".join(aligned2), best_score, (curr_i, curr_j), (m, best_j), cells_computed

    @staticmethod
    def _generate_cigar(aligned1: str, aligned2: str) -> str:
        """Generate standard CIGAR string (e.g. '18M2D2M') from aligned strings."""
        if not aligned1:
            return ""

        ops = []
        for c1, c2 in zip(aligned1, aligned2):
            if c1 == "-":
                ops.append("I")  # Insertion relative to seq1
            elif c2 == "-":
                ops.append("D")  # Deletion relative to seq1
            else:
                ops.append("M")  # Match or Mismatch

        # Run-length encode
        cigar_parts = []
        current_op = ops[0]
        count = 1
        for op in ops[1:]:
            if op == current_op:
                count += 1
            else:
                cigar_parts.append(f"{count}{current_op}")
                current_op = op
                count = 1
        cigar_parts.append(f"{count}{current_op}")
        return "".join(cigar_parts)


# Aliases for specific standard algorithms
def needleman_wunsch(
    seq1: str,
    seq2: str,
    match_score: int = 2,
    mismatch_penalty: int = -1,
    gap_penalty: int = -2,
) -> AlignmentResult:
    """Convenience function for Needleman-Wunsch global alignment."""
    aligner = SequenceAligner(match_score, mismatch_penalty, gap_penalty)
    return aligner.align(seq1, seq2, mode=AlignmentType.GLOBAL)


def smith_waterman(
    seq1: str,
    seq2: str,
    match_score: int = 2,
    mismatch_penalty: int = -1,
    gap_penalty: int = -2,
) -> AlignmentResult:
    """Convenience function for Smith-Waterman local alignment."""
    aligner = SequenceAligner(match_score, mismatch_penalty, gap_penalty)
    return aligner.align(seq1, seq2, mode=AlignmentType.LOCAL)

