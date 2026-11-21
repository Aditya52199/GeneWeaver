"""
Numba @cuda.jit kernels for GPU-accelerated sequence alignment.

Implements:
1. Anti-diagonal parallel dynamic programming kernels (Needleman-Wunsch & Smith-Waterman).
2. Massively parallel batch CRISPR guide-target scanning kernel.
3. Fast parallel PAM motif search kernel.
"""

from cuda_engine.device_manager import DeviceManager
DeviceManager.initialize()

from numba import cuda
import numpy as np

from cuda_engine.sequence_encoding import (
    gpu_is_spcas9_pam,
    gpu_max3,
    gpu_max4,
    gpu_score,
)

# Traceback direction constants
TB_STOP = 0
TB_DIAG = 1
TB_UP = 2
TB_LEFT = 3


# =========================================================================
# 1. Anti-Diagonal Needleman-Wunsch Kernel
# =========================================================================

@cuda.jit
def needleman_wunsch_diag_kernel(
    d_seq1,
    d_seq2,
    d_matrix,
    d_traceback,
    diag_k,
    start_i,
    end_i,
    match_score,
    mismatch_penalty,
    gap_penalty,
):
    """
    Compute all independent cells along a single anti-diagonal (i + j = diag_k) in parallel.

    Args:
        d_seq1: 1D device array of encoded query sequence.
        d_seq2: 1D device array of encoded target sequence.
        d_matrix: 2D device array for score DP matrix of shape (m+1, n+1).
        d_traceback: 2D device array for traceback directions (m+1, n+1).
        diag_k: Current anti-diagonal index (i + j).
        start_i: Minimum row index on this anti-diagonal.
        end_i: Maximum row index on this anti-diagonal.
        match_score: Score for match.
        mismatch_penalty: Penalty for mismatch.
        gap_penalty: Penalty for gap.
    """
    tid = cuda.grid(1)
    i = start_i + tid

    if i <= end_i:
        j = diag_k - i

        b1 = d_seq1[i - 1]
        b2 = d_seq2[j - 1]

        score_sub = gpu_score(b1, b2, match_score, mismatch_penalty)
        diag_val = d_matrix[i - 1, j - 1] + score_sub
        up_val = d_matrix[i - 1, j] + gap_penalty
        left_val = d_matrix[i, j - 1] + gap_penalty

        best_score = diag_val
        best_dir = TB_DIAG

        if up_val > best_score:
            best_score = up_val
            best_dir = TB_UP

        if left_val > best_score:
            best_score = left_val
            best_dir = TB_LEFT

        d_matrix[i, j] = best_score
        d_traceback[i, j] = best_dir


# =========================================================================
# 2. Anti-Diagonal Smith-Waterman Local Alignment Kernel
# =========================================================================

@cuda.jit
def smith_waterman_diag_kernel(
    d_seq1,
    d_seq2,
    d_matrix,
    d_traceback,
    d_max_tracker,
    diag_k,
    start_i,
    end_i,
    match_score,
    mismatch_penalty,
    gap_penalty,
):
    """
    Compute local alignment anti-diagonal in parallel, maintaining non-negative scores.
    d_max_tracker: 1D array of shape (3,) storing [max_score, max_i, max_j].
    """
    tid = cuda.grid(1)
    i = start_i + tid

    if i <= end_i:
        j = diag_k - i

        b1 = d_seq1[i - 1]
        b2 = d_seq2[j - 1]

        score_sub = gpu_score(b1, b2, match_score, mismatch_penalty)
        diag_val = d_matrix[i - 1, j - 1] + score_sub
        up_val = d_matrix[i - 1, j] + gap_penalty
        left_val = d_matrix[i, j - 1] + gap_penalty

        best_score = 0
        best_dir = TB_STOP

        if diag_val > best_score:
            best_score = diag_val
            best_dir = TB_DIAG
        if up_val > best_score:
            best_score = up_val
            best_dir = TB_UP
        if left_val > best_score:
            best_score = left_val
            best_dir = TB_LEFT

        d_matrix[i, j] = best_score
        d_traceback[i, j] = best_dir

        if best_score > 0:
            cuda.atomic.max(d_max_tracker, 0, best_score)


# =========================================================================
# 3. Batch CRISPR Off-Target & PAM Scanning Kernel
# =========================================================================

@cuda.jit
def batch_crispr_scan_kernel(
    d_target_chunk,      # 1D array of target chunk (e.g. 10,000 bp)
    d_guide,             # 1D array of guide RNA (e.g. 23 bp: 20bp spacer + 3bp PAM)
    d_hit_positions,     # 1D output array for hit start coordinates
    d_hit_scores,        # 1D output array for alignment scores
    d_hit_mismatches,    # 1D output array for mismatch count
    d_hit_count,         # 1D array of shape (1,) for atomic hit counter
    max_hits,            # Maximum hits to record
    max_mismatches,      # e.g., 4 allowable mismatches
    match_score,
    mismatch_penalty,
):
    """
    Massively parallel CRISPR off-target scanner.
    Each GPU thread inspects one sliding window across the genomic target chunk.
    """
    target_len = d_target_chunk.shape[0]
    guide_len = d_guide.shape[0]  # typically 23 (20 spacer + 3 PAM)

    idx = cuda.grid(1)

    # Valid start indices that fit full guide
    if idx <= target_len - guide_len:
        # Check SpCas9 PAM: target[idx + 21] == 'G' and target[idx + 22] == 'G'
        # 'G' is encoded as 2
        pam_mid = d_target_chunk[idx + 21]
        pam_last = d_target_chunk[idx + 22]

        if gpu_is_spcas9_pam(pam_mid, pam_last):
            # Compute mismatches and score across the 20bp spacer
            mismatches = 0
            score = 0

            for p in range(20):
                t_base = d_target_chunk[idx + p]
                g_base = d_guide[p]

                if t_base == g_base and t_base != 4:  # 4 is 'N'
                    score += match_score
                else:
                    mismatches += 1
                    score += mismatch_penalty

            # Award score for matching PAM
            score += 2 * match_score

            if mismatches <= max_mismatches:
                out_idx = cuda.atomic.add(d_hit_count, 0, 1)
                if out_idx < max_hits:
                    d_hit_positions[out_idx] = idx
                    d_hit_scores[out_idx] = score
                    d_hit_mismatches[out_idx] = mismatches


# =========================================================================
# 4. Fast Parallel PAM Locator Kernel
# =========================================================================

@cuda.jit
def parallel_pam_finder_kernel(
    d_target_chunk,
    d_pam_indices,
    d_pam_count,
    max_pam_records,
):
    """
    Scan genomic array in parallel across all threads to locate all 'NGG' PAM sites.
    """
    idx = cuda.grid(1)
    target_len = d_target_chunk.shape[0]

    # Inspect position idx as the 'N' of 'NGG'
    if idx <= target_len - 3:
        b_mid = d_target_chunk[idx + 1]
        b_last = d_target_chunk[idx + 2]

        if gpu_is_spcas9_pam(b_mid, b_last):
            out_pos = cuda.atomic.add(d_pam_count, 0, 1)
            if out_pos < max_pam_records:
                d_pam_indices[out_pos] = idx
