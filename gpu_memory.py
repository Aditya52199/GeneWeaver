
import gc
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional


@dataclass
class MemoryPlan:
    total_windows: int
    batch_size: int
    number_of_batches: int
    estimated_memory_per_batch_bytes: int
    available_gpu_memory_bytes: Optional[int]
    memory_source: str


class GPUMemoryManager:
    """
    Plans bounded batches for genomic alignment.

    Memory estimates cover the main CUDA input/output arrays.
    They are estimates, not guarantees of total CUDA runtime usage.
    """

    def __init__(
        self,
        max_batch_starts: int = 65536,
        memory_fraction: float = 0.60
    ):
        if max_batch_starts < 1:
            raise ValueError("max_batch_starts must be at least 1.")

        if not 0 < memory_fraction <= 1:
            raise ValueError("memory_fraction must be between 0 and 1.")

        self.max_batch_starts = max_batch_starts
        self.memory_fraction = memory_fraction

    @staticmethod
    def estimate_batch_memory(
        sequence_length: int,
        batch_size: int
    ) -> int:
        """
        Approximate main device-array memory:

        sequence: 1 byte per base
        guide: 20 bytes
        start indices: 4 bytes per window
        mismatch counts: 2 bytes per window
        mismatch masks: 4 bytes per window
        """
        sequence_bytes = sequence_length
        guide_bytes = 20
        per_window_bytes = 4 + 2 + 4

        return (
            sequence_bytes
            + guide_bytes
            + (batch_size * per_window_bytes)
        )

    @staticmethod
    def get_available_gpu_memory(cuda_module):
        """
        Return free CUDA memory in bytes.

        Returns None if the runtime does not expose memory information,
        including some simulator configurations.
        """
        try:
            context = cuda_module.current_context()
            memory_info = context.get_memory_info()
            return int(memory_info[0])
        except Exception:
            return None

    def plan(
        self,
        sequence_length: int,
        total_windows: int,
        cuda_module=None
    ) -> MemoryPlan:

        if sequence_length < 0 or total_windows < 0:
            raise ValueError("Sequence length and window count cannot be negative.")

        if total_windows == 0:
            return MemoryPlan(
                total_windows=0,
                batch_size=0,
                number_of_batches=0,
                estimated_memory_per_batch_bytes=0,
                available_gpu_memory_bytes=None,
                memory_source="No windows"
            )

        batch_size = min(
            total_windows,
            self.max_batch_starts
        )

        available_memory = None
        memory_source = "Configured batch limit"

        if cuda_module is not None:
            available_memory = self.get_available_gpu_memory(
                cuda_module
            )

            if available_memory is not None:
                usable_memory = int(
                    available_memory * self.memory_fraction
                )

                fixed_memory = sequence_length + 20
                bytes_per_window = 10

                memory_limited_batch = (
                    usable_memory - fixed_memory
                ) // bytes_per_window

                if memory_limited_batch < 1:
                    raise MemoryError(
                        "Insufficient reported CUDA memory for even one "
                        "alignment window. Reduce sequence/chunk size."
                    )

                batch_size = min(
                    batch_size,
                    memory_limited_batch
                )

                memory_source = "CUDA free-memory estimate"

        number_of_batches = (
            total_windows + batch_size - 1
        ) // batch_size

        estimated_memory = self.estimate_batch_memory(
            sequence_length,
            batch_size
        )

        return MemoryPlan(
            total_windows=total_windows,
            batch_size=batch_size,
            number_of_batches=number_of_batches,
            estimated_memory_per_batch_bytes=estimated_memory,
            available_gpu_memory_bytes=available_memory,
            memory_source=memory_source
        )

    @staticmethod
    def iter_batches(
        start_indices: Iterable[int],
        batch_size: int
    ) -> Iterator[List[int]]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        batch = []

        for start in start_indices:
            batch.append(int(start))

            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    @staticmethod
    def release_host_references():
        """
        Collect temporary Python references after a batch.

        CUDA device buffers are released when their references
        are dropped and the CUDA runtime processes deallocations.
        """
        gc.collect()
