# GeneWeaver CPU Baseline Performance Report

## System Information
- **Python Version**: 3.14.8
- **Platform**: win32

## Baseline Metrics Overview

### Algorithm: `GLOBAL` (Needleman-Wunsch)
- **Peak MCUPS**: 4.0817
- **Average MCUPS**: 3.7947

| Seq1 (bp) | Seq2 (bp) | Matrix Cells | Avg Time (ms) | Std Dev (ms) | MCUPS | Throughput (bp/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 50 | 50 | 2,601 | 0.637 | 0.115 | 4.0817 | 156,928.4 |
| 100 | 101 | 10,302 | 2.923 | 0.160 | 3.5243 | 68,762.6 |
| 250 | 249 | 62,750 | 15.444 | 1.478 | 4.0629 | 32,309.2 |
| 500 | 500 | 251,001 | 67.853 | 6.735 | 3.6992 | 14,737.7 |
| 1000 | 995 | 996,996 | 276.527 | 5.115 | 3.6054 | 7,214.5 |

### Algorithm: `LOCAL` (Smith-Waterman)
- **Peak MCUPS**: 3.9024
- **Average MCUPS**: 3.4666

| Seq1 (bp) | Seq2 (bp) | Matrix Cells | Avg Time (ms) | Std Dev (ms) | MCUPS | Throughput (bp/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 50 | 50 | 2,601 | 0.732 | 0.083 | 3.5528 | 136,593.4 |
| 100 | 101 | 10,302 | 2.640 | 0.032 | 3.9024 | 76,138.3 |
| 250 | 249 | 62,750 | 16.957 | 0.551 | 3.7006 | 29,427.8 |
| 500 | 500 | 251,001 | 84.350 | 13.699 | 2.9757 | 11,855.4 |
| 1000 | 995 | 996,996 | 311.420 | 11.287 | 3.2015 | 6,406.1 |

### Algorithm: `CRISPR_SEMI_GLOBAL`
- **Peak MCUPS**: 4.8511
- **Average MCUPS**: 4.0480

| Seq1 (bp) | Seq2 (bp) | Matrix Cells | Avg Time (ms) | Std Dev (ms) | MCUPS | Throughput (bp/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 20 | 23 | 504 | 0.172 | 0.005 | 2.9251 | 249,564.7 |
| 20 | 100 | 2,121 | 0.551 | 0.076 | 3.8477 | 217,693.7 |
| 20 | 500 | 10,521 | 2.169 | 0.094 | 4.8511 | 239,763.9 |
| 20 | 1000 | 21,021 | 4.602 | 0.176 | 4.5683 | 221,665.2 |

## Key Baseline Takeaways
1. **Quadratic Scaling**: Single-core CPU dynamic programming displays expected $O(m \times n)$ scaling. For sequence lengths of 1,000 bp (~1,000,000 cells), alignment takes ~276ms - 311ms.
2. **Cell Update Rate**: The pure Python engine delivers ~3.5 to 4.8 MCUPS (Million Cell Updates Per Second).
3. **CRISPR gRNA Target Windows**: A 20bp guide aligned against a 1kb locus executes in ~4.6ms.
4. **Acceleration Target**: The upcoming CUDA GPU engine should target $\ge 100\times$ speedup by computing anti-diagonals in parallel and executing batch off-target evaluations concurrently.

