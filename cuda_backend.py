
import os

import numpy as np

from geneweaver import (
    WINDOW_LENGTH,
    GUIDE_LENGTH,
    GenomicChunk,
    reverse_complement,
    scan_chunk,
    calculate_gc_content,
    calculate_candidate_score,
    classify_priority,
)

try:
    from numba import cuda

    NUMBA_CUDA_IMPORTED = True
    CUDA_AVAILABLE = cuda.is_available()
    CUDA_IMPORT_ERROR = None

except Exception as error:
    cuda = None
    NUMBA_CUDA_IMPORTED = False
    CUDA_AVAILABLE = False
    CUDA_IMPORT_ERROR = str(error)


THREADS_PER_BLOCK = 128


if NUMBA_CUDA_IMPORTED:

    @cuda.jit
    def alignment_kernel(
        sequence,
        guide,
        start_indices,
        maximum_mismatches,
        output_counts,
        output_masks
    ):
        """
        One CUDA thread processes one possible 23-base target window.

        output_counts:
            -1 = PAM not found
            0 or greater = mismatch count

        output_masks:
            Bit mask indicating mismatch positions.
        """

        thread_id = cuda.grid(1)

        if thread_id >= start_indices.size:
            return

        start = start_indices[thread_id]

        # Check the SpCas9 NGG PAM.
        pam_first = sequence[start + GUIDE_LENGTH]
        pam_second = sequence[start + GUIDE_LENGTH + 1]
        pam_third = sequence[start + GUIDE_LENGTH + 2]

        # The first PAM position must be a valid DNA base.
        if (
            pam_first != 65
            and pam_first != 67
            and pam_first != 71
            and pam_first != 84
        ):
            output_counts[thread_id] = -1
            output_masks[thread_id] = 0
            return

        if pam_second != 71 or pam_third != 71:
            output_counts[thread_id] = -1
            output_masks[thread_id] = 0
            return

        mismatch_count = 0
        mismatch_mask = 0

        for position in range(GUIDE_LENGTH):
            if sequence[start + position] != guide[position]:
                mismatch_count += 1
                mismatch_mask |= (1 << position)

                # Stop comparing as soon as the limit is exceeded.
                if mismatch_count > maximum_mismatches:
                    break

        output_counts[thread_id] = mismatch_count
        output_masks[thread_id] = mismatch_mask


def _build_candidate(
    oriented_sequence,
    start,
    chunk,
    strand,
    mismatch_count,
    mismatch_mask
):
    target = oriented_sequence[start:start + GUIDE_LENGTH]
    pam = oriented_sequence[
        start + GUIDE_LENGTH:start + WINDOW_LENGTH
    ]

    mismatch_positions = [
        position + 1
        for position in range(GUIDE_LENGTH)
        if mismatch_mask & (1 << position)
    ]

    seed_mismatches = sum(
        1 for position in mismatch_positions
        if position >= 11
    )

    if strand == "+":
        genomic_start = chunk.start_position + start
    else:
        genomic_start = (
            chunk.start_position
            + len(oriented_sequence)
            - WINDOW_LENGTH
            - start
        )

    score = calculate_candidate_score(
        mismatch_count,
        seed_mismatches
    )

    return {
        "chromosome": chunk.chromosome,
        "start_position": genomic_start,
        "end_position": genomic_start + WINDOW_LENGTH - 1,
        "strand": strand,
        "guide": None,
        "target": target.decode("ascii"),
        "pam": pam.decode("ascii"),
        "mismatch_count": mismatch_count,
        "mismatch_positions": ",".join(
            map(str, mismatch_positions)
        ),
        "seed_mismatches": seed_mismatches,
        "gc_content": calculate_gc_content(target),
        "candidate_score": score,
        "priority": classify_priority(score),
        "chunk_index": chunk.chunk_index
    }


def _scan_orientation_cuda(
    oriented_sequence,
    start_indices,
    chunk,
    strand,
    guide,
    maximum_mismatches
):
    if len(start_indices) == 0:
        return [], 0

    sequence_array = np.frombuffer(
        oriented_sequence,
        dtype=np.uint8
    ).copy()

    guide_array = np.frombuffer(
        guide,
        dtype=np.uint8
    ).copy()

    starts_array = np.asarray(
        start_indices,
        dtype=np.int32
    )

    output_counts = np.full(
        len(start_indices),
        -1,
        dtype=np.int16
    )

    output_masks = np.zeros(
        len(start_indices),
        dtype=np.uint32
    )

    device_sequence = cuda.to_device(sequence_array)
    device_guide = cuda.to_device(guide_array)
    device_starts = cuda.to_device(starts_array)

    device_counts = cuda.to_device(output_counts)
    device_masks = cuda.to_device(output_masks)

    blocks = (
        len(start_indices) + THREADS_PER_BLOCK - 1
    ) // THREADS_PER_BLOCK

    alignment_kernel[
        blocks,
        THREADS_PER_BLOCK
    ](
        device_sequence,
        device_guide,
        device_starts,
        maximum_mismatches,
        device_counts,
        device_masks
    )

    cuda.synchronize()

    result_counts = device_counts.copy_to_host()
    result_masks = device_masks.copy_to_host()

    candidates = []
    valid_pam_sites = 0

    for index, start in enumerate(start_indices):
        mismatch_count = int(result_counts[index])

        if mismatch_count < 0:
            continue

        valid_pam_sites += 1

        if mismatch_count > maximum_mismatches:
            continue

        candidate = _build_candidate(
            oriented_sequence=oriented_sequence,
            start=start,
            chunk=chunk,
            strand=strand,
            mismatch_count=mismatch_count,
            mismatch_mask=int(result_masks[index])
        )

        candidate["guide"] = guide.decode("ascii")
        candidates.append(candidate)

    return candidates, valid_pam_sites


def scan_chunk_accelerated(
    chunk: GenomicChunk,
    guide: bytes,
    maximum_mismatches: int
):
    """
    Use CUDA when available.

    Otherwise, automatically use the existing CPU scan_chunk()
    implementation.
    """

    if not CUDA_AVAILABLE:
        candidates, statistics = scan_chunk(
            chunk,
            guide,
            maximum_mismatches
        )

        return {
            "backend": "CPU",
            "candidates": candidates,
            "statistics": statistics,
            "message": (
                "CUDA unavailable. Existing CPU alignment "
                "implementation was used."
            )
        }

    sequence = chunk.sequence

    total_starts = max(
        0,
        len(sequence) - WINDOW_LENGTH + 1
    )

    valid_starts = min(
        chunk.valid_starts,
        total_starts
    )

    forward_indices = list(range(valid_starts))

    reverse_first_index = total_starts - valid_starts
    reverse_indices = list(
        range(reverse_first_index, total_starts)
    )

    forward_candidates, forward_pam_count = (
        _scan_orientation_cuda(
            oriented_sequence=sequence,
            start_indices=forward_indices,
            chunk=chunk,
            strand="+",
            guide=guide,
            maximum_mismatches=maximum_mismatches
        )
    )

    reverse_sequence = reverse_complement(sequence)

    reverse_candidates, reverse_pam_count = (
        _scan_orientation_cuda(
            oriented_sequence=reverse_sequence,
            start_indices=reverse_indices,
            chunk=chunk,
            strand="-",
            guide=guide,
            maximum_mismatches=maximum_mismatches
        )
    )

    candidates = forward_candidates + reverse_candidates

    statistics = {
        "chunk_index": chunk.chunk_index,
        "sequence_length": len(sequence),
        "valid_starts": chunk.valid_starts,
        "pam_sites": forward_pam_count + reverse_pam_count,
        "candidates": len(candidates)
    }

    return {
        "backend": "CUDA",
        "candidates": candidates,
        "statistics": statistics,
        "message": "Numba CUDA alignment kernel executed."
    }


def get_backend_status():
    return {
        "numba_cuda_imported": NUMBA_CUDA_IMPORTED,
        "cuda_available": CUDA_AVAILABLE,
        "execution_backend": (
            "CUDA" if CUDA_AVAILABLE else "CPU"
        ),
        "cuda_import_error": CUDA_IMPORT_ERROR,
        "simulator_enabled": (
            os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
        )
    }


if __name__ == "__main__":
    import json

    print(
        json.dumps(
            get_backend_status(),
            indent=4
        )
    )
