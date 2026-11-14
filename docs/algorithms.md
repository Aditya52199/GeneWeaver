# GeneWeaver: Sequence Alignment Algorithms

This document describes the DNA sequence alignment algorithms implemented in GeneWeaver.

## Implemented Algorithms

### 1. Needleman-Wunsch (Global Alignment)
- **Use Case**: End-to-end alignment of two sequences of similar length.
- **Complexity**: Time $O(m \times n)$, Space $O(m \times n)$.
- **Recurrence**:
  $$H(i, j) = \max \begin{cases} H(i-1, j-1) + S(seq_1[i-1], seq_2[j-1]) \\ H(i-1, j) + \gamma \\ H(i, j-1) + \gamma \end{cases}$$
- **Module**: `alignment.sequence_aligner.needleman_wunsch`

### 2. Smith-Waterman (Local Alignment)
- **Use Case**: Finding high-scoring local matching subsequences.
- **Complexity**: Time $O(m \times n)$, Space $O(m \times n)$.
- **Recurrence**:
  $$H(i, j) = \max \begin{cases} 0 \\ H(i-1, j-1) + S(seq_1[i-1], seq_2[j-1]) \\ H(i-1, j) + \gamma \\ H(i, j-1) + \gamma \end{cases}$$
- **Module**: `alignment.sequence_aligner.smith_waterman`

### 3. Semi-Global Alignment (CRISPR Guide RNA Matching)
- **Use Case**: Aligning a short 20bp guide RNA + PAM (23bp) against a long genomic window (e.g. 100bp - 10,000bp) with free start and end gaps on the target sequence.
- **Module**: `alignment.sequence_aligner.SequenceAligner(mode=AlignmentType.SEMI_GLOBAL)`

## Performance Metrics
- **MCUPS (Mega Cell Updates Per Second)**: $\frac{m \times n}{\text{Execution Time (s)} \times 10^6}$
- **Throughput**: $\frac{\text{Total Base Pairs}}{\text{Execution Time (s)}}$

