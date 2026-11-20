# GeneWeaver: GPU-Accelerated CRISPR Alignment Engine

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Numba%20%40cuda.jit-76B900.svg?logo=nvidia)](https://numba.pydata.org/)
[![Tests](https://img.shields.io/badge/Tests-24%20Passing-brightgreen.svg)](tests/)
[![BioPython](https://img.shields.io/badge/BioPython-1.88-orange.svg)](https://biopython.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**GeneWeaver** is a high-performance bioinformatics engine designed to accelerate DNA sequence alignment and genome-wide CRISPR/Cas9 off-target search. By combining **BioPython** streaming genomic pipelines with **Numba `@cuda.jit`** GPU dynamic programming kernels and explicit **Host (CPU RAM) $\leftrightarrow$ Device (GPU VRAM)** memory orchestration, GeneWeaver bridges the gap between biological accuracy and hardware-level throughput.

---

## Architecture Overview

```
                                      +-----------------------------------------+
                                      |   Reference Genome (FASTA / FASTA.GZ)   |
                                      +-----------------------------------------+
                                                           |
                                                           v
                                         [ BioPython Genomic Stream Reader ]
                                         - Upper/lower case repeat normalization
                                         - Ambiguous base (N) masking & validation
                                                           |
                                                           v
                                         [ Overlapping NumPy Array Chunker ]
                                         - 10,000 bp chunks (dtype: uint8 ASCII)
                                         - 100 bp boundary overlap (PAM guard)
                                                           |
                 +-----------------------------------------+-----------------------------------------+
                 |                                                                                   |
                 v                                                                                   v
    +-------------------------+                                                         +-------------------------+
    |    CPU Baseline Engine  |                                                         |    CUDA Engine (GPU)    |
    |  - Needleman-Wunsch     |                                                         |  - Pinned Host RAM      |
    |  - Smith-Waterman       |                                                         |  - Host -> Device DMA   |
    |  - Semi-Global (CRISPR) |                                                         |  - Anti-Diagonal JIT    |
    |  - Metrics: MCUPS, bp/s |                                                         |  - Parallel Batch Scan  |
    +-------------------------+                                                         |  - Device -> Host DMA   |
                 |                                                                      +-------------------------+
                 v                                                                                   |
    +-------------------------+                                                                      v
    |  Baseline Verification  | <================ Comparison & Benchmarking ================> +-------------------------+
    |  (Exact Score Standard) |                                                                | High-Throughput Targets |
    +-------------------------+                                                                +-------------------------+
```

---

## Key Features

### 1. Pure Python CPU Baseline
- **Needleman-Wunsch (Global)**: Classical $O(m \times n)$ global dynamic programming with full traceback.
- **Smith-Waterman (Local)**: Local alignment with non-negative score boundaries and traceback from peak cells.
- **Semi-Global (CRISPR Guide RNA)**: End-free target gaps tailored for aligning short guide RNAs (20bp spacer + 3bp PAM) across large genomic loci.
- **Standard Metrics**: Alignment score, CIGAR string, identity percentage, execution time (ms), and **MCUPS** (Mega Cell Updates Per Second).

### 2. BioPython Genomic Data Pipeline
- **Streaming FASTA Reader**: Memory-efficient ingestion of multi-gigabyte chromosomes via `Bio.SeqIO.parse`.
- **Gzip Compression**: Direct transparent reading of `.fasta.gz` archives without pre-decompression.
- **Sequence Sanitization**: Normalizes soft-masked repetitive sequences (`acgt` $\rightarrow$ `ACGT`) and filters non-canonical characters.
- **Boundary-Safe Array Chunker**: Slices chromosomes into contiguous 1D NumPy arrays (`uint8` ASCII byte layout) with configurable overlap (e.g. 100bp) to ensure boundary PAM sites are never missed.

### 3. Numba `@cuda.jit` GPU Engine
- **Anti-Diagonal Wavefront Dynamic Programming**:
  Parallelizes independent cells along anti-diagonals $k = i + j$, utilizing GPU thread blocks for cell matrix updates.
- **Massively Parallel Batch CRISPR Scanner**:
  Dispatches sliding window evaluations across thousands of candidate loci concurrently, verifying `NGG` PAM motifs and computing mismatch penalties in GPU registers.
- **Device Functions (`@cuda.jit(device=True)`)**:
  In-kernel nucleotide scoring, branchless maximum selection, and SpCas9 PAM verification.

### 4. Explicit Host $\leftrightarrow$ Device Memory Management
- **Pinned Host Memory**: Allocates page-locked host memory (`cuda.pinned_array`) for high-bandwidth Direct Memory Access (DMA).
- **GPU VRAM Buffer Allocation**: Manages device arrays (`cuda.device_array`) for scores, tracebacks, and hit counters.
- **Telemetry & Bandwidth Profiling**: Measures latency (ms) and throughput (GB/s) for both Host-to-Device and Device-to-Host transfers.
- **Hybrid Portability**: Automatically runs on hardware NVIDIA GPUs (e.g. RTX 4050, Compute Capability 8.9) or falls back to Numba's CUDA Simulator (`CUDASIM`) for CI/CD and systems without full CUDA compiler toolchains.

---

## Directory Structure

```text
GeneWeaver/
├── alignment/                   # Alignment algorithms and data structures
│   ├── __init__.py
│   ├── sequence_aligner.py      # Pure Python NW, SW, and Semi-Global aligners
│   └── guide_rna.py             # Guide RNA definitions and utilities
├── benchmarks/                  # Performance benchmarks and reports
│   ├── __init__.py
│   ├── cpu_benchmark.py         # CPU baseline scaling benchmark suite
│   ├── gpu_benchmark.py         # GPU vs CPU benchmark & transfer timings
│   ├── benchmark_report.py      # Markdown report generator
│   └── baseline_metrics.json    # Recorded reference benchmark metrics
├── cuda_engine/                 # Numba @cuda.jit GPU acceleration
│   ├── __init__.py
│   ├── device_manager.py        # Device querying, VRAM stats, CUDASIM fallback
│   ├── memory_manager.py        # Host (RAM) <-> Device (VRAM) memory manager
│   ├── sequence_encoding.py     # DNA byte encoding & @cuda.jit(device=True) helpers
│   ├── kernels.py               # Numba @cuda.jit anti-diagonal & batch kernels
│   ├── gpu_matcher.py           # High-level CUDASyncAligner & CUDABatchCRISPRScanner
│   └── cpu_fallback.py          # Fallback handler when GPU is unavailable
├── genomic_parser/              # BioPython ingestion and chunking pipeline
│   ├── __init__.py
│   ├── fasta_reader.py          # BioPython streaming reader (.fasta & .fasta.gz)
│   ├── sequence_chunker.py      # NumPy array chunker with overlap padding
│   ├── sequence_cleaner.py      # Sequence sanitization and repeat normalization
│   ├── sequence_validator.py    # Composition analysis and GC-content calculator
│   └── metadata_extractor.py    # Chromosome and genome statistics
├── scripts/                     # Utility and diagnostic scripts
│   ├── check_cuda.py            # GPU hardware & runtime diagnostics
│   └── generate_test_data.py    # Realistic mock human genome FASTA generator
├── tests/                       # Unit test suites (24 tests)
│   ├── test_alignment.py        # CPU aligner and benchmark tests
│   ├── test_fasta_reader.py     # BioPython parser and chunker tests
│   └── test_cuda_kernels.py     # CUDA kernels and memory transfer tests
├── data/                        # Sample and mock genomic FASTA files
├── docs/                        # Technical documentation & algorithm specs
│   ├── algorithms.md            # Detailed algorithm recurrence formulas
│   └── benchmarking.md          # Baseline performance reports
├── main.py                      # CLI entry point for genomic pipeline demo
├── pyproject.toml
└── requirements.txt
```

---

## Quickstart

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Aditya52199/GeneWeaver.git
cd GeneWeaver

pip install -r requirements.txt
```

### 2. Verify GPU & CUDA Environment

Inspect active GPU hardware, driver support, and VRAM:

```bash
python scripts/check_cuda.py
```

*Sample output:*
```text
============================================================
 GeneWeaver CUDA Diagnostic Utility
============================================================
[+] CUDA Runtime Initialized: True
[+] Mode: Hardware CUDA Execution (or CUDASIM)
[+] Active Device Information:
    - Name               : NVIDIA GeForce RTX 4050 Laptop GPU
    - Compute Capability : 8.9
    - Total VRAM         : 6,144.0 MB (6,442,450,944 bytes)
============================================================
```

### 3. Generate Mock Human Genome

Generate a realistic mock human genome FASTA with SpCas9 PAM sites, ~41% GC content, and soft-masked repeats:

```bash
# Generate uncompressed FASTA
python scripts/generate_test_data.py --output data/mock_human_genome.fasta

# Generate compressed FASTA (.gz)
python scripts/generate_test_data.py --output data/mock_human_genome.fasta.gz --gzip
```

### 4. Run the Genomic Ingestion & Chunking Pipeline

Parse the genome with BioPython and slice it into contiguous NumPy arrays:

```bash
python main.py --fasta data/mock_human_genome.fasta --chunk-size 10000 --overlap 100
```

*Output summary:*
```text
======================================================================
 GeneWeaver: Genomic Ingestion and Array Chunking Pipeline
======================================================================
[1] Parsing FASTA using BioPython: data/mock_human_genome.fasta
    - Total Genome Length : 160,000 bp
    - Chromosomes Loaded  : 4 (chr1, chr2, chr21, chrX)
    - Overall GC Content  : 40.90%

[2] Chunking into NumPy Arrays (chunk_size=10,000 bp, overlap=100 bp)...
    - Total Chunks Created: 19
    - Pipeline Time       : 17.94 ms
    - Processing Rate     : ~8.9M bp/sec
```

---

## Performance Benchmarks

### CPU Baseline (Pure Python 3.14 on Windows)

Measurements recorded across $N=3$ trials per length setting:

| Algorithm | Sequence Length (bp) | Matrix Cells ($m \times n$) | Mean Time (ms) | Peak MCUPS | Throughput (bp/s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Needleman-Wunsch** | 50 × 50 | 2,601 | **0.637 ms** | 4.0817 | 156,928 bp/s |
| **Needleman-Wunsch** | 100 × 101 | 10,302 | **2.923 ms** | 3.5243 | 68,763 bp/s |
| **Needleman-Wunsch** | 250 × 249 | 62,750 | **15.444 ms** | 4.0629 | 32,309 bp/s |
| **Needleman-Wunsch** | 500 × 500 | 251,001 | **67.853 ms** | 3.6992 | 14,738 bp/s |
| **Needleman-Wunsch** | 1,000 × 995 | 996,996 | **276.527 ms** | 3.6054 | 7,215 bp/s |
| **Smith-Waterman** | 1,000 × 995 | 996,996 | **311.420 ms** | 3.2015 | 6,406 bp/s |
| **CRISPR Semi-Global** | 20bp × 1,000bp | 21,021 | **4.602 ms** | 4.5683 | 221,665 bp/s |

Run the full benchmark suite:

```bash
python benchmarks/cpu_benchmark.py --lengths 50 100 250 500 1000 --trials 3
```

### GPU vs CPU Comparison & Transfer Timings

Run the comparison benchmark:

```bash
python benchmarks/gpu_benchmark.py --lengths 50 100 200 --trials 2
```

Host-to-Device and Device-to-Host transfer latency remains sub-millisecond ($<0.50\text{ ms}$), enabling rapid data interchange between host preprocessing and device kernels.

---

## Running Unit Tests

The test suite contains **24 tests** validating sequence alignment correctness, BioPython parsing, and GPU memory lifecycle:

```bash
python -m pytest tests/ -v
```

*Test breakdown:*
- `tests/test_alignment.py` (11 tests): Global, local, semi-global score identity, gap penalties, CIGAR generation, and edge cases.
- `tests/test_fasta_reader.py` (7 tests): BioPython FASTA stream reading, `.gz` decompression, repeat sanitization, and array chunking.
- `tests/test_cuda_kernels.py` (6 tests): Host $\leftrightarrow$ Device transfers, pinned memory allocation, `@cuda.jit` diagonal kernels, and batch CRISPR scanning.

---

## Next Steps & Roadmap

- [x] CPU Baseline sequence aligner (Needleman-Wunsch, Smith-Waterman, Semi-Global).
- [x] BioPython genomic streaming pipeline with overlapping NumPy chunking.
- [x] Numba `@cuda.jit` anti-diagonal dynamic programming and batch CRISPR kernels.
- [x] Host-Device memory management and bandwidth telemetry.
- [ ] Multi-GPU scaling and Dask distributed dispatcher (`scheduler/chunk_dispatcher.py`).
- [ ] Real-time monitoring dashboard for GPU VRAM and alignment throughput (`dashboard/`).
- [ ] SQLite/PostgreSQL result persistence (`storage/result_repository.py`).

---

## License

This project is licensed under the MIT License.
