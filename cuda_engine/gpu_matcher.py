"""
GPU Sequence Matcher and CRISPR Off-Target Engine for GeneWeaver.

Orchestrates Host-Device memory transfers, kernel launches,
anti-diagonal synchronization, and off-target candidate collection.
"""

from dataclasses import dataclass
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from alignment.sequence_aligner import AlignmentResult, AlignmentType
from cuda_engine.device_manager import DeviceManager
DeviceManager.initialize()

from cuda_engine.kernels import (
    TB_DIAG,
    TB_LEFT,
    TB_STOP,
    TB_UP,
    batch_crispr_scan_kernel,
    needleman_wunsch_diag_kernel,
    parallel_pam_finder_kernel,
    smith_waterman_diag_kernel,
)
from cuda_engine.memory_manager import MemoryManager, TransferStats
from cuda_engine.sequence_encoding import decode_sequence, encode_sequence


@dataclass
class OffTargetHit:
    """Represents a candidate off-target site detected by GPU."""
    position: int
    score: int
    mismatches: int
    sequence: str
    pam: str


class CUDASyncAligner:
    """
    GPU-accelerated DNA sequence aligner using Numba @cuda.jit kernels
    and anti-diagonal parallel dynamic programming.
    """

    def __init__(
        self,
        match_score: int = 2,
        mismatch_penalty: int = -1,
        gap_penalty: int = -2,
        threads_per_block: int = 128,
    ):
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty
        self.threads_per_block = threads_per_block
        self.memory_manager = MemoryManager()

    def align(
        self,
        seq1: str,
        seq2: str,
        mode: AlignmentType = AlignmentType.GLOBAL,
    ) -> AlignmentResult:
        """
        Align two sequences on GPU handling full Host <-> Device memory transfers.

        Args:
            seq1: First sequence (query).
            seq2: Second sequence (target).
            mode: GLOBAL, LOCAL, or SEMI_GLOBAL (free target-end gaps).

        Returns:
            AlignmentResult with timing, score, and alignment strings.
        """
        seq1 = seq1.strip().upper()
        seq2 = seq2.strip().upper()
        if mode not in (
            AlignmentType.GLOBAL,
            AlignmentType.LOCAL,
            AlignmentType.SEMI_GLOBAL,
        ):
            raise ValueError(f"Unsupported alignment mode: {mode}")

        m = len(seq1)
        n = len(seq2)

        total_start = time.perf_counter()

        # Step 1: Host Encoding
        h_seq1 = encode_sequence(seq1)
        h_seq2 = encode_sequence(seq2)

        # Step 2: Host -> Device Memory Transfer
        d_seq1 = self.memory_manager.to_device(h_seq1)
        d_seq2 = self.memory_manager.to_device(h_seq2)

        # Allocate DP score matrix and traceback matrix on GPU VRAM
        # Step 3: Initialize Boundaries on GPU
        # Initialize row 0 and col 0 on host, then copy or initialize
        h_init_matrix = np.zeros((m + 1, n + 1), dtype=np.int32)
        h_init_tb = np.zeros((m + 1, n + 1), dtype=np.uint8)

        if mode in (AlignmentType.GLOBAL, AlignmentType.SEMI_GLOBAL):
            for i in range(1, m + 1):
                h_init_matrix[i, 0] = i * self.gap_penalty
                h_init_tb[i, 0] = TB_UP

        if mode == AlignmentType.GLOBAL:
            for j in range(1, n + 1):
                h_init_matrix[0, j] = j * self.gap_penalty
                h_init_tb[0, j] = TB_LEFT

        # Transfer initial boundaries to Device
        d_matrix = self.memory_manager.to_device(h_init_matrix)
        d_traceback = self.memory_manager.to_device(h_init_tb)

        d_max_tracker = None
        if mode == AlignmentType.LOCAL:
            d_max_tracker = self.memory_manager.to_device(np.zeros(3, dtype=np.int32))

        # Step 4: Anti-Diagonal Wavefront GPU Execution
        # Diagonals k range from 2 to m + n
        total_diagonals = m + n
        for diag_k in range(2, total_diagonals + 1):
            start_i = max(1, diag_k - n)
            end_i = min(m, diag_k - 1)
            num_cells = end_i - start_i + 1

            if num_cells <= 0:
                continue

            blocks = (num_cells + self.threads_per_block - 1) // self.threads_per_block

            if mode in (AlignmentType.GLOBAL, AlignmentType.SEMI_GLOBAL):
                needleman_wunsch_diag_kernel[blocks, self.threads_per_block](
                    d_seq1,
                    d_seq2,
                    d_matrix,
                    d_traceback,
                    diag_k,
                    start_i,
                    end_i,
                    self.match_score,
                    self.mismatch_penalty,
                    self.gap_penalty,
                )
            elif mode == AlignmentType.LOCAL:
                smith_waterman_diag_kernel[blocks, self.threads_per_block](
                    d_seq1,
                    d_seq2,
                    d_matrix,
                    d_traceback,
                    d_max_tracker,
                    diag_k,
                    start_i,
                    end_i,
                    self.match_score,
                    self.mismatch_penalty,
                    self.gap_penalty,
                )

        DeviceManager.synchronize()

        # Step 5: Device -> Host Memory Transfer
        h_matrix = self.memory_manager.to_host(d_matrix)
        h_traceback = self.memory_manager.to_host(d_traceback)

        total_elapsed = time.perf_counter() - total_start

        # Step 6: Host Traceback and Result Assembly
        if mode == AlignmentType.GLOBAL:
            final_score = int(h_matrix[m, n])
            aligned1, aligned2, start_pos, end_pos = self._traceback_global(
                seq1, seq2, h_traceback, m, n
            )
        elif mode == AlignmentType.SEMI_GLOBAL:
            final_score, best_j = self._semi_global_endpoint(h_matrix, m, n)
            aligned1, aligned2, start_pos, end_pos = self._traceback_semi_global(
                seq1, seq2, h_traceback, m, best_j
            )
        else:
            # Find maximum score coordinates for local alignment
            max_idx = np.unravel_index(np.argmax(h_matrix), h_matrix.shape)
            final_score = int(h_matrix[max_idx])
            aligned1, aligned2, start_pos, end_pos = self._traceback_local(
                seq1, seq2, h_matrix, h_traceback, max_idx[0], max_idx[1]
            )

        # Alignment statistics
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

        total_aligned = len(aligned1)
        identity = (matches / total_aligned * 100.0) if total_aligned > 0 else 0.0
        cells_computed = (m + 1) * (n + 1)
        mcups = (cells_computed / (total_elapsed * 1e6)) if total_elapsed > 0 else 0.0

        cigar = self._cigar(aligned1, aligned2)

        return AlignmentResult(
            seq1_aligned=aligned1,
            seq2_aligned=aligned2,
            score=final_score,
            alignment_type=f"cuda_{mode.value if isinstance(mode, AlignmentType) else str(mode)}",
            execution_time_seconds=total_elapsed,
            seq1_length=m,
            seq2_length=n,
            cells_computed=cells_computed,
            mcups=round(mcups, 4),
            start_pos=start_pos,
            end_pos=end_pos,
            matches=matches,
            mismatches=mismatches,
            gaps=gaps,
            identity_percent=round(identity, 2),
            cigar=cigar,
        )

    @staticmethod
    def _traceback_global(
        seq1: str, seq2: str, tb_matrix: np.ndarray, m: int, n: int
    ) -> Tuple[str, str, Tuple[int, int], Tuple[int, int]]:
        """Reconstruct global alignment from Device-transferred traceback matrix."""
        aligned1 = []
        aligned2 = []
        curr_i = m
        curr_j = n

        while curr_i > 0 or curr_j > 0:
            direction = tb_matrix[curr_i, curr_j]
            if direction == TB_DIAG:
                aligned1.append(seq1[curr_i - 1])
                aligned2.append(seq2[curr_j - 1])
                curr_i -= 1
                curr_j -= 1
            elif direction == TB_UP:
                aligned1.append(seq1[curr_i - 1])
                aligned2.append("-")
                curr_i -= 1
            elif direction == TB_LEFT:
                aligned1.append("-")
                aligned2.append(seq2[curr_j - 1])
                curr_j -= 1
            else:
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
        return "".join(aligned1), "".join(aligned2), (0, 0), (m, n)

    @staticmethod
    def _semi_global_endpoint(
        score_matrix: np.ndarray, m: int, n: int
    ) -> Tuple[int, int]:
        """Choose the highest-scoring cell on the final query row."""
        best_j = n
        best_score = int(score_matrix[m, n])
        for j in range(1, n + 1):
            score = int(score_matrix[m, j])
            if score > best_score:
                best_score = score
                best_j = j
        return best_score, best_j

    @staticmethod
    def _traceback_semi_global(
        seq1: str, seq2: str, tb_matrix: np.ndarray, m: int, best_j: int
    ) -> Tuple[str, str, Tuple[int, int], Tuple[int, int]]:
        """Trace back from the best final-row cell, leaving target ends free."""
        aligned1 = []
        aligned2 = []
        curr_i = m
        curr_j = best_j

        while curr_i > 0 and curr_j > 0:
            direction = tb_matrix[curr_i, curr_j]
            if direction == TB_DIAG:
                aligned1.append(seq1[curr_i - 1])
                aligned2.append(seq2[curr_j - 1])
                curr_i -= 1
                curr_j -= 1
            elif direction == TB_UP:
                aligned1.append(seq1[curr_i - 1])
                aligned2.append("-")
                curr_i -= 1
            elif direction == TB_LEFT:
                aligned1.append("-")
                aligned2.append(seq2[curr_j - 1])
                curr_j -= 1
            else:
                break

        while curr_i > 0:
            aligned1.append(seq1[curr_i - 1])
            aligned2.append("-")
            curr_i -= 1

        aligned1.reverse()
        aligned2.reverse()
        return (
            "".join(aligned1),
            "".join(aligned2),
            (curr_i, curr_j),
            (m, best_j),
        )

    @staticmethod
    def _traceback_local(
        seq1: str, seq2: str, score_matrix: np.ndarray, tb_matrix: np.ndarray, max_i: int, max_j: int
    ) -> Tuple[str, str, Tuple[int, int], Tuple[int, int]]:
        """Reconstruct local alignment from peak cell until score <= 0."""
        aligned1 = []
        aligned2 = []
        curr_i = max_i
        curr_j = max_j

        while curr_i > 0 and curr_j > 0 and score_matrix[curr_i, curr_j] > 0:
            direction = tb_matrix[curr_i, curr_j]
            if direction == TB_DIAG:
                aligned1.append(seq1[curr_i - 1])
                aligned2.append(seq2[curr_j - 1])
                curr_i -= 1
                curr_j -= 1
            elif direction == TB_UP:
                aligned1.append(seq1[curr_i - 1])
                aligned2.append("-")
                curr_i -= 1
            elif direction == TB_LEFT:
                aligned1.append("-")
                aligned2.append(seq2[curr_j - 1])
                curr_j -= 1
            else:
                break

        aligned1.reverse()
        aligned2.reverse()
        return "".join(aligned1), "".join(aligned2), (curr_i, curr_j), (max_i, max_j)

    @staticmethod
    def _cigar(aligned1: str, aligned2: str) -> str:
        """Construct CIGAR string."""
        if not aligned1:
            return ""
        ops = []
        for c1, c2 in zip(aligned1, aligned2):
            if c1 == "-":
                ops.append("I")
            elif c2 == "-":
                ops.append("D")
            else:
                ops.append("M")

        cigar_parts = []
        curr_op = ops[0]
        cnt = 1
        for op in ops[1:]:
            if op == curr_op:
                cnt += 1
            else:
                cigar_parts.append(f"{cnt}{curr_op}")
                curr_op = op
                cnt = 1
        cigar_parts.append(f"{cnt}{curr_op}")
        return "".join(cigar_parts)


class CUDABatchCRISPRScanner:
    """
    Massively parallel CRISPR off-target scanner on GPU.
    Dispatches thousands of genomic window evaluations concurrently to GPU threads.
    """

    def __init__(
        self,
        match_score: int = 2,
        mismatch_penalty: int = -1,
        max_mismatches: int = 4,
        max_hits: int = 1000,
        threads_per_block: int = 256,
    ):
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.max_mismatches = max_mismatches
        self.max_hits = max_hits
        self.threads_per_block = threads_per_block
        self.memory_manager = MemoryManager()

    def scan_chunk(
        self,
        target_chunk_seq: str,
        guide_seq: str,
    ) -> List[OffTargetHit]:
        """
        Scan a genomic sequence chunk against a CRISPR guide RNA on GPU.

        Args:
            target_chunk_seq: Target DNA sequence (e.g. 10,000 bp).
            guide_seq: 23bp guide sequence (20bp spacer + 3bp PAM, e.g. '...NGG').

        Returns:
            List of OffTargetHit candidates found on the GPU.
        """
        guide_seq = guide_seq.strip().upper()
        target_chunk_seq = target_chunk_seq.strip().upper()

        if len(guide_seq) < 23:
            raise ValueError(f"Guide RNA sequence must be at least 23bp (got {len(guide_seq)}bp)")

        # 1. Host encoding
        h_target = encode_sequence(target_chunk_seq)
        h_guide = encode_sequence(guide_seq[:23])

        # 2. Host -> Device Memory Transfer
        d_target = self.memory_manager.to_device(h_target)
        d_guide = self.memory_manager.to_device(h_guide)

        # Allocate output hit buffers on GPU VRAM
        d_hit_positions = self.memory_manager.allocate_device(self.max_hits, dtype=np.int32)
        d_hit_scores = self.memory_manager.allocate_device(self.max_hits, dtype=np.int32)
        d_hit_mismatches = self.memory_manager.allocate_device(self.max_hits, dtype=np.int32)
        d_hit_count = self.memory_manager.to_device(np.zeros(1, dtype=np.int32))

        # 3. Launch GPU Kernel
        num_windows = len(h_target) - len(h_guide) + 1
        if num_windows <= 0:
            return []

        blocks = (num_windows + self.threads_per_block - 1) // self.threads_per_block

        batch_crispr_scan_kernel[blocks, self.threads_per_block](
            d_target,
            d_guide,
            d_hit_positions,
            d_hit_scores,
            d_hit_mismatches,
            d_hit_count,
            self.max_hits,
            self.max_mismatches,
            self.match_score,
            self.mismatch_penalty,
        )
        DeviceManager.synchronize()

        # 4. Device -> Host Memory Transfer
        h_count = self.memory_manager.to_host(d_hit_count)[0]
        actual_hits = min(int(h_count), self.max_hits)

        if actual_hits == 0:
            return []

        # Slice and transfer hit arrays
        h_positions = self.memory_manager.to_host(d_hit_positions)[:actual_hits]
        h_scores = self.memory_manager.to_host(d_hit_scores)[:actual_hits]
        h_mismatches = self.memory_manager.to_host(d_hit_mismatches)[:actual_hits]

        hits = []
        for i in range(actual_hits):
            pos = int(h_positions[i])
            hit_subseq = target_chunk_seq[pos : pos + 23]
            pam = hit_subseq[20:23]
            hits.append(
                OffTargetHit(
                    position=pos,
                    score=int(h_scores[i]),
                    mismatches=int(h_mismatches[i]),
                    sequence=hit_subseq,
                    pam=pam,
                )
            )

        return hits
