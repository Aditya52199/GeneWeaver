"""
Sequence encoding and GPU device helper functions.

Provides:
- 2-bit/byte numeric encoding of DNA nucleotides ('A':0, 'C':1, 'G':2, 'T':3, 'N':4)
- @cuda.jit(device=True) functions for in-kernel scoring, max selection, and PAM detection.
"""

import numpy as np
from cuda_engine.device_manager import DeviceManager

DeviceManager.initialize()
from numba import cuda

# Host-side encoding lookup tables
_ENCODE_DICT = {
    ord("A"): 0, ord("a"): 0,
    ord("C"): 1, ord("c"): 1,
    ord("G"): 2, ord("g"): 2,
    ord("T"): 3, ord("t"): 3,
    ord("N"): 4, ord("n"): 4,
}
_DECODE_CHARS = ["A", "C", "G", "T", "N"]


def encode_sequence(seq: str) -> np.ndarray:
    """
    Encode DNA string into a 1D NumPy array of uint8 values (0-4).

    Args:
        seq: DNA sequence string.

    Returns:
        np.ndarray of dtype uint8.
    """
    encoded = np.empty(len(seq), dtype=np.uint8)
    for i, char in enumerate(seq):
        encoded[i] = _ENCODE_DICT.get(ord(char), 4)
    return encoded


def decode_sequence(arr: np.ndarray) -> str:
    """Decode a 1D uint8 encoded array back to a DNA string."""
    return "".join(_DECODE_CHARS[val] if val < 5 else "N" for val in arr)


# =========================================================================
# CUDA Device Functions (Callable ONLY inside @cuda.jit kernels)
# =========================================================================

@cuda.jit(device=True)
def gpu_score(b1: int, b2: int, match_score: int, mismatch_penalty: int) -> int:
    """
    Compute substitution score on GPU.
    Base 4 is 'N' (ambiguous), which scores as a mismatch.
    """
    if b1 == 4 or b2 == 4:
        return mismatch_penalty
    if b1 == b2:
        return match_score
    return mismatch_penalty


@cuda.jit(device=True)
def gpu_max3(a: int, b: int, c: int) -> int:
    """Compute maximum of three integer values."""
    m = a
    if b > m:
        m = b
    if c > m:
        m = c
    return m


@cuda.jit(device=True)
def gpu_max4(a: int, b: int, c: int, d: int) -> int:
    """Compute maximum of four integer values (e.g. for Smith-Waterman with 0 bound)."""
    m = a
    if b > m:
        m = b
    if c > m:
        m = c
    if d > m:
        m = d
    return m


@cuda.jit(device=True)
def gpu_is_spcas9_pam(b_mid: int, b_last: int) -> bool:
    """
    Check if a candidate PAM is 'NGG' (SpCas9).
    Where 'G' is encoded as 2.
    """
    return (b_mid == 2) and (b_last == 2)
