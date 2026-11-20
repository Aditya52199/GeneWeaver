"""
CPU fallback aligner when GPU is not available.
"""

from alignment.sequence_aligner import AlignmentResult, AlignmentType, SequenceAligner


class CPUFallbackAligner:
    """Seamless CPU fallback for GPU alignment pipelines."""

    def __init__(self, match_score: int = 2, mismatch_penalty: int = -1, gap_penalty: int = -2):
        self._aligner = SequenceAligner(
            match_score=match_score,
            mismatch_penalty=mismatch_penalty,
            gap_penalty=gap_penalty,
        )

    def align(self, seq1: str, seq2: str, mode: AlignmentType = AlignmentType.GLOBAL) -> AlignmentResult:
        res = self._aligner.align(seq1, seq2, mode=mode)
        # Note fallback in result
        return AlignmentResult(
            seq1_aligned=res.seq1_aligned,
            seq2_aligned=res.seq2_aligned,
            score=res.score,
            alignment_type=f"cpu_fallback_{res.alignment_type}",
            execution_time_seconds=res.execution_time_seconds,
            seq1_length=res.seq1_length,
            seq2_length=res.seq2_length,
            cells_computed=res.cells_computed,
            mcups=res.mcups,
            start_pos=res.start_pos,
            end_pos=res.end_pos,
            matches=res.matches,
            mismatches=res.mismatches,
            gaps=res.gaps,
            identity_percent=res.identity_percent,
            cigar=res.cigar,
        )
