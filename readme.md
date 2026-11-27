# GeneWeaver

GeneWeaver is an academic research tool for finding PAM-compatible sequence candidates similar to a CRISPR guide. It includes a streaming Biopython FASTA pipeline, an auditable CPU Hamming reference, optional Numba CUDA kernels, optional Dask CPU distribution, and CSV/TSV/JSON output.

**A reported candidate is a computational prediction, not an experimentally confirmed mutation or a clinical risk estimate.** The bundled ranking score is an explicitly labeled heuristic, not CFD.

## Install

```powershell
python -m pip install -e .
python -m pip install -e ".[gpu]"          # optional Numba CUDA
python -m pip install -e ".[distributed]"  # optional Dask
python -m pip install -e ".[dashboard]"    # optional Textual
python -m pip install -e ".[dev]"          # pytest
```

CUDA execution requires compatible NVIDIA hardware, drivers, and Numba CUDA runtime support. CPU-only installations remain functional. See [installation](docs/INSTALLATION.md) and [CUDA notes](docs/CUDA.md).

## Use

```powershell
python scripts/generate_sample_genome.py
python -m geneweaver.main scan --fasta data/sample/sample_genome.fasta --guide ATCGGCTAACGTAGCTACGA --pam NGG --max-mismatches 4 --engine cpu --output results.csv --progress
python -m geneweaver.main scan --fasta data/sample/sample_genome.fasta --guide ATCGGCTAACGTAGCTACGA --engine cuda --cuda-kernel shared --output results.json
python -m geneweaver.main scan --fasta data/sample/sample_genome.fasta --guide ATCGGCTAACGTAGCTACGA --engine distributed
# For GPU Dask workers, pass --scheduler-address and explicit --gpu-worker WORKER=DEVICE assignments.
python -m geneweaver.main benchmark --length 100000
python -m geneweaver.main dashboard
python -m geneweaver.main --help
```

Unavailable optional engines report their dependency or device status; CUDA selection falls back to CPU and includes the fallback reason in its scan report. The CLI `gpu-info` reports current device discovery. Coordinates are zero-based, half-open. Ambiguous reference symbols are normalized to N and cannot match guide bases.

## Architecture

See [architecture diagram](docs/ARCHITECTURE.md), [algorithms and scoring](docs/ALGORITHMS.md), [Dask scheduling](docs/DASK.md), and [benchmark methodology](docs/BENCHMARKS.md).

CUDA kernels execute independently on the GPU while Python prepares data and launches work; this does not mean every part of the application bypasses the GIL. Benchmark values are measured at runtime, and speedup is workload and hardware dependent.

## Validate

`python -m pytest` runs the test suite when the optional dev dependency is installed. `python scripts/check_environment.py` checks available integrations. GPU and Dask hardware/worker validations must be run on a machine that has those resources.
