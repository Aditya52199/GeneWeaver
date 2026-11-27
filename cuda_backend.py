
import gc
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

from gpu_memory import GPUMemoryManager


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
DEFAULT_MAX_BATCH_STARTS = 65536

MEMORY_MANAGER = GPUMemoryManager(
    max_batch_starts=DEFAULT_MAX_BATCH_STARTS,
    memory_fraction=0.60
)


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
        thread_id = cuda.grid(1)

        if thread_id >= start_indices.size:
            return

        start = start_indices[thread_id]

        # Check the canonical SpCas9 NGG PAM.
        pam_first = sequence[start + GUIDE_LENGTH]
        pam_second = sequence[start + GUIDE_LENGTH + 1]
        pam_third = sequence[start + GUIDE_LENGTH + 2]

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

                if mismatch_count > maximum_mismatches:
                    break

        output_counts[thread_id] = mismatch_count
        output_masks[thread_id] = mismatch_mask


def _build_candidate(
    oriented_sequence,
    start,
    chunk,
    strand,
    guide,
    mismatch_count,
    mismatch_mask
):
    target = oriented_sequence[
        start:start + GUIDE_LENGTH
    ]

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

    candidate_sequence = target.decode("ascii")
    similarity = 1.0 - mismatch_count / GUIDE_LENGTH
    predicted_score = max(0.0, 1.0 - 0.16 * mismatch_count - 0.11 * seed_mismatches)

    return {
        "chromosome": chunk.chromosome,
        "start": genomic_start,
        "end": genomic_start + WINDOW_LENGTH,
        "start_position": genomic_start,
        "end_position": genomic_start + WINDOW_LENGTH - 1,
        "strand": strand,
        "guide": guide.decode("ascii"),
        "target": candidate_sequence,
        "candidate": candidate_sequence,
        "pam": pam.decode("ascii"),
        "mismatch_count": mismatch_count,
        "mismatch_positions": ",".join(
            map(str, mismatch_positions)
        ),
        "mismatch_positions_list": mismatch_positions,
        "seed_mismatches": seed_mismatches,
        "gc_content": calculate_gc_content(target),
        "candidate_score": score,
        "priority": classify_priority(score),
        "chunk_index": chunk.chunk_index,
        "sequence_similarity": similarity,
        "predicted_off_target_score": round(predicted_score, 6),
        "score_model": "GeneWeaver heuristic v1 (not CFD)",
        "engine": "cuda",
        "experimentally_validated": False
    }


def _execute_cuda_batch(
    sequence_array,
    guide_array,
    start_batch,
    maximum_mismatches
):
    """
    Allocate, execute and copy back one bounded CUDA batch.

    Local device-array references are released when this function
    returns, allowing the runtime to reclaim temporary buffers.
    """
    starts_array = np.asarray(
        start_batch,
        dtype=np.int32
    )

    output_counts = np.full(
        len(start_batch),
        -1,
        dtype=np.int16
    )

    output_masks = np.zeros(
        len(start_batch),
        dtype=np.uint32
    )

    device_sequence = cuda.to_device(sequence_array)
    device_guide = cuda.to_device(guide_array)
    device_starts = cuda.to_device(starts_array)

    device_counts = cuda.to_device(output_counts)
    device_masks = cuda.to_device(output_masks)

    blocks = (
        len(start_batch) + THREADS_PER_BLOCK - 1
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

    # Drop references to temporary device buffers.
    del device_sequence
    del device_guide
    del device_starts
    del device_counts
    del device_masks

    return result_counts, result_masks


def _scan_orientation_cuda(
    oriented_sequence,
    start_indices,
    chunk,
    strand,
    guide,
    maximum_mismatches,
    max_batch_starts=DEFAULT_MAX_BATCH_STARTS
):
    total_windows = len(start_indices)

    if total_windows == 0:
        return [], 0, {
            "batch_count": 0,
            "batch_size": 0,
            "estimated_memory_bytes": 0
        }

    manager = GPUMemoryManager(
        max_batch_starts=max_batch_starts,
        memory_fraction=0.60
    )

    memory_plan = manager.plan(
        sequence_length=len(oriented_sequence),
        total_windows=total_windows,
        cuda_module=cuda
    )

    sequence_array = np.frombuffer(
        oriented_sequence,
        dtype=np.uint8
    ).copy()

    guide_array = np.frombuffer(
        guide,
        dtype=np.uint8
    ).copy()

    candidates = []
    valid_pam_sites = 0
    batches_processed = 0

    for start_batch in manager.iter_batches(
        start_indices,
        memory_plan.batch_size
    ):
        result_counts, result_masks = _execute_cuda_batch(
            sequence_array=sequence_array,
            guide_array=guide_array,
            start_batch=start_batch,
            maximum_mismatches=maximum_mismatches
        )

        batches_processed += 1

        for index, start in enumerate(start_batch):
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
                guide=guide,
                mismatch_count=mismatch_count,
                mismatch_mask=int(result_masks[index])
            )

            candidates.append(candidate)

        del result_counts
        del result_masks

        manager.release_host_references()

    memory_statistics = {
        "batch_count": batches_processed,
        "batch_size": memory_plan.batch_size,
        "estimated_memory_bytes": (
            memory_plan.estimated_memory_per_batch_bytes
        )
    }

    return candidates, valid_pam_sites, memory_statistics


def scan_chunk_accelerated(
    chunk: GenomicChunk,
    guide: bytes,
    maximum_mismatches: int,
    max_batch_starts=DEFAULT_MAX_BATCH_STARTS
):
    """
    Run bounded CUDA batches when CUDA is available.
    Otherwise, use the existing CPU alignment implementation.
    """

    if not CUDA_AVAILABLE:
        candidates, statistics = scan_chunk(
            chunk,
            guide,
            maximum_mismatches
        )

        statistics["batch_count"] = 1
        statistics["batch_size"] = chunk.valid_starts

        return {
            "backend": "CPU",
            "candidates": candidates,
            "statistics": statistics,
            "memory": {
                "batch_count": 1,
                "batch_size": chunk.valid_starts,
                "estimated_memory_bytes": None
            },
            "message": (
                "CUDA unavailable. Existing CPU alignment "
                "implementation was used."
            )
        }

    if len(guide) != GUIDE_LENGTH:
        raise ValueError("Guide must contain exactly 20 bases.")

    if any(base not in b"ACGT" for base in guide):
        raise ValueError("Guide must contain only A, C, G and T.")

    if not 0 <= maximum_mismatches <= GUIDE_LENGTH:
        raise ValueError("Invalid maximum mismatch count.")

    total_starts = max(
        0,
        len(chunk.sequence) - WINDOW_LENGTH + 1
    )

    valid_starts = min(
        chunk.valid_starts,
        total_starts
    )

    forward_indices = range(valid_starts)

    reverse_first_index = total_starts - valid_starts
    reverse_indices = range(
        reverse_first_index,
        total_starts
    )

    forward_candidates, forward_pam_count, forward_memory = (
        _scan_orientation_cuda(
            oriented_sequence=chunk.sequence,
            start_indices=forward_indices,
            chunk=chunk,
            strand="+",
            guide=guide,
            maximum_mismatches=maximum_mismatches,
            max_batch_starts=max_batch_starts
        )
    )

    reverse_sequence = reverse_complement(chunk.sequence)

    reverse_candidates, reverse_pam_count, reverse_memory = (
        _scan_orientation_cuda(
            oriented_sequence=reverse_sequence,
            start_indices=reverse_indices,
            chunk=chunk,
            strand="-",
            guide=guide,
            maximum_mismatches=maximum_mismatches,
            max_batch_starts=max_batch_starts
        )
    )

    candidates = forward_candidates + reverse_candidates

    statistics = {
        "chunk_index": chunk.chunk_index,
        "sequence_length": len(chunk.sequence),
        "valid_starts": chunk.valid_starts,
        "pam_sites": forward_pam_count + reverse_pam_count,
        "candidates": len(candidates),
        "batch_count": (
            forward_memory["batch_count"]
            + reverse_memory["batch_count"]
        )
    }

    memory = {
        "forward": forward_memory,
        "reverse": reverse_memory
    }

    return {
        "backend": "CUDA",
        "candidates": candidates,
        "statistics": statistics,
        "memory": memory,
        "message": (
            "CUDA alignment completed using bounded batches."
        )
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
        ),
        "maximum_batch_starts": DEFAULT_MAX_BATCH_STARTS
    }


if __name__ == "__main__":
    import json

    print(json.dumps(get_backend_status(), indent=4))
