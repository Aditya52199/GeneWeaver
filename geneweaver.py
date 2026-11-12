
"""
====================================================================
GENEWEAVER - GPU-ACCELERATED CRISPR ALIGNMENT ENGINE
====================================================================

COMMIT 2: ADVANCED GENOMIC PREPROCESSING AND STREAMING

Features:
1. Streaming FASTA parser
2. Bounded-memory genomic chunk processing
3. Multiple FASTA chromosome support
4. DNA normalization and validation
5. Ambiguous nucleotide detection
6. Overlapping genomic chunks
7. SpCas9 NGG PAM recognition
8. Forward and reverse strand alignment
9. Mismatch position analysis
10. PAM-proximal seed mismatch analysis
11. GC-content calculation
12. Computational candidate scoring
13. Duplicate prevention
14. CPU performance statistics
15. Detailed CSV export

Coordinates: Zero-based, end-exclusive.

DISCLAIMER:
Candidate scores are computational screening priorities,
not experimentally validated biological risk probabilities.
====================================================================
"""

import argparse
import csv
import logging
import re
import sys
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple


# ====================================================================
# 1. CONFIGURATION
# ====================================================================

APP_NAME = "GeneWeaver"
VERSION = "0.2.0"

GUIDE_LENGTH = 20
PAM_LENGTH = 3
WINDOW_LENGTH = GUIDE_LENGTH + PAM_LENGTH

DEFAULT_CHUNK_SIZE = 1_000_000
DEFAULT_MAX_MISMATCHES = 3

# PAM-proximal seed region in the 20-base protospacer.
SEED_START_POSITION = 11

VALID_BASES = set("ACGT")


# ====================================================================
# 2. LOGGING
# ====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(APP_NAME)


# ====================================================================
# 3. DATA STRUCTURES
# ====================================================================

@dataclass
class GenomeChunk:
    chromosome: str
    sequence: str
    start_position: int
    end_position: int

    # Number of starting positions owned by this chunk.
    # Remaining positions belong to the next overlapping chunk.
    valid_starts: int


@dataclass
class OffTargetResult:
    chromosome: str
    start_position: int
    end_position: int
    strand: str
    target_sequence: str
    pam_sequence: str
    mismatch_count: int
    mismatch_positions: str
    seed_mismatches: int
    gc_content: float
    candidate_score: float
    priority: str


@dataclass
class ScanStatistics:
    total_bases: int = 0
    ambiguous_bases: int = 0
    total_chunks: int = 0
    total_windows: int = 0
    valid_pam_sites: int = 0
    candidate_sites: int = 0
    execution_time: float = 0.0


# ====================================================================
# 4. DNA UTILITIES
# ====================================================================

def reverse_complement(sequence: str) -> str:

    complement = str.maketrans(
        "ACGTN",
        "TGCAN"
    )

    return sequence.translate(complement)[::-1]


def validate_guide_rna(guide: str) -> str:

    guide = guide.strip().upper().replace("U", "T")

    if len(guide) != GUIDE_LENGTH:

        raise ValueError(
            f"Guide must contain exactly {GUIDE_LENGTH} bases."
        )

    invalid = set(guide) - VALID_BASES

    if invalid:

        raise ValueError(
            f"Invalid guide bases: {invalid}"
        )

    return guide


def clean_dna_sequence(sequence: str) -> str:
    """
    Normalize DNA sequence.

    Invalid characters are converted to N.
    Coordinates are preserved.
    """

    sequence = sequence.upper().replace("U", "T")

    sequence = re.sub(r"\s+", "", sequence)

    return "".join(
        base if base in VALID_BASES else "N"
        for base in sequence
    )


def calculate_gc_content(sequence: str) -> float:
    """
    Calculate GC percentage.

    Ambiguous bases are excluded from the denominator.
    """

    valid_bases = [
        base for base in sequence
        if base in VALID_BASES
    ]

    if not valid_bases:
        return 0.0

    gc_count = sum(
        base in ("G", "C")
        for base in valid_bases
    )

    return round(
        gc_count / len(valid_bases) * 100,
        2
    )


# ====================================================================
# 5. STREAMING FASTA PARSER
# ====================================================================

class GenomicParser:
    """
    Memory-conscious FASTA parser.

    Reads sequence lines progressively instead of loading
    an entire chromosome into memory.

    Supports:
    - Multiple chromosome records
    - Leading comments
    - Blank lines
    - UTF-8 BOM
    - Lowercase sequences
    - Ambiguous nucleotides
    - Overlapping chunks
    """

    def __init__(self, fasta_path: str):

        self.fasta_path = Path(fasta_path)

        if not self.fasta_path.exists():

            raise FileNotFoundError(
                f"FASTA file not found: {self.fasta_path}"
            )

        if not self.fasta_path.is_file():

            raise ValueError(
                "Provided path is not a file."
            )

        if self.fasta_path.stat().st_size == 0:

            raise ValueError(
                "FASTA file is empty."
            )

        self.total_bases = 0
        self.ambiguous_bases = 0
        self.record_count = 0

    def parse_chunks(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE
    ) -> Iterator[GenomeChunk]:
        """
        Stream FASTA records directly into genomic chunks.

        Each full chunk retains WINDOW_LENGTH - 1 bases
        as overlap for the next chunk.

        Every genomic window is assigned to exactly one
        chunk using valid_starts.
        """

        if chunk_size < WINDOW_LENGTH:

            raise ValueError(
                f"Chunk size must be at least {WINDOW_LENGTH}."
            )

        overlap = WINDOW_LENGTH - 1

        step = chunk_size - overlap

        chromosome = None

        buffer = ""

        chunk_start = 0

        def emit_final_chunk():

            nonlocal buffer, chunk_start

            if len(buffer) < WINDOW_LENGTH:
                return None

            valid_starts = (
                len(buffer) - WINDOW_LENGTH + 1
            )

            chunk = GenomeChunk(
                chromosome=chromosome,
                sequence=buffer,
                start_position=chunk_start,
                end_position=chunk_start + len(buffer),
                valid_starts=valid_starts
            )

            return chunk

        logger.info(
            "Starting streaming FASTA preprocessing: %s",
            self.fasta_path
        )

        try:

            with self.fasta_path.open(
                "r",
                encoding="utf-8-sig"
            ) as fasta_file:

                for line_number, line in enumerate(
                    fasta_file,
                    start=1
                ):

                    line = line.strip()

                    if not line:
                        continue

                    # Ignore comments.
                    if line.startswith(("#", ";", "!")):
                        continue

                    # New chromosome record.
                    if line.startswith(">"):

                        # Flush the previous chromosome.
                        if chromosome is not None:

                            final_chunk = emit_final_chunk()

                            if final_chunk is not None:
                                yield final_chunk

                        header = line[1:].strip()

                        if not header:

                            raise ValueError(
                                f"Empty FASTA header at line {line_number}."
                            )

                        chromosome = header.split()[0]

                        self.record_count += 1

                        logger.info(
                            "Processing chromosome: %s",
                            chromosome
                        )

                        buffer = ""

                        chunk_start = 0

                        continue

                    # Ignore text before the first FASTA header.
                    if chromosome is None:
                        continue

                    cleaned = clean_dna_sequence(line)

                    self.total_bases += len(cleaned)

                    self.ambiguous_bases += sum(
                        base == "N"
                        for base in cleaned
                    )

                    buffer += cleaned

                    # Emit chunks whenever enough sequence is available.
                    while len(buffer) >= chunk_size:

                        chunk = GenomeChunk(
                            chromosome=chromosome,
                            sequence=buffer[:chunk_size],
                            start_position=chunk_start,
                            end_position=chunk_start + chunk_size,
                            valid_starts=step
                        )

                        yield chunk

                        # Preserve overlap for the next chunk.
                        buffer = buffer[step:]

                        chunk_start += step

            # Flush the final chromosome.
            if chromosome is not None:

                final_chunk = emit_final_chunk()

                if final_chunk is not None:
                    yield final_chunk

            if self.record_count == 0:

                raise ValueError(
                    "No valid FASTA records found."
                )

        except UnicodeDecodeError as error:

            raise ValueError(
                "Unable to decode FASTA file. "
                "Use a UTF-8 encoded FASTA file."
            ) from error


# ====================================================================
# 6. CPU ALIGNMENT ENGINE
# ====================================================================

class CPUAlignmentEngine:
    """
    CPU baseline CRISPR alignment engine.

    Searches:
        20-base protospacer + NGG PAM

    Both genomic orientations are examined.
    """

    def __init__(
        self,
        guide_sequence: str,
        max_mismatches: int = DEFAULT_MAX_MISMATCHES
    ):

        self.guide = validate_guide_rna(
            guide_sequence
        )

        if not 0 <= max_mismatches <= GUIDE_LENGTH:

            raise ValueError(
                "Maximum mismatches must be between 0 and 20."
            )

        self.max_mismatches = max_mismatches

        self.statistics = ScanStatistics()

        self.results: List[OffTargetResult] = []

    # ----------------------------------------------------------------
    # PAM VALIDATION
    # ----------------------------------------------------------------

    @staticmethod
    def is_valid_pam(pam: str) -> bool:

        return (
            len(pam) == PAM_LENGTH
            and all(base in VALID_BASES for base in pam)
            and pam[1:] == "GG"
        )

    # ----------------------------------------------------------------
    # MISMATCH ANALYSIS
    # ----------------------------------------------------------------

    def calculate_mismatches(
        self,
        target: str
    ) -> Tuple[int, List[int]]:

        positions = []

        for index in range(GUIDE_LENGTH):

            if self.guide[index] != target[index]:

                positions.append(index + 1)

        return len(positions), positions

    # ----------------------------------------------------------------
    # SEED REGION ANALYSIS
    # ----------------------------------------------------------------

    @staticmethod
    def calculate_seed_mismatches(
        mismatch_positions: List[int]
    ) -> int:
        """
        Count mismatches in guide positions 11-20.

        This represents the PAM-proximal seed region
        under the orientation convention used here.
        """

        return sum(
            position >= SEED_START_POSITION
            for position in mismatch_positions
        )

    # ----------------------------------------------------------------
    # CANDIDATE SCORING
    # ----------------------------------------------------------------

    @staticmethod
    def calculate_candidate_score(
        mismatch_count: int,
        seed_mismatches: int
    ) -> float:
        """
        Heuristic computational priority score.

        This is not a biological probability.
        """

        score = (
            100
            - (mismatch_count * 15)
            - (seed_mismatches * 5)
        )

        return round(
            max(0.0, min(100.0, score)),
            2
        )

    @staticmethod
    def classify_candidate(score: float) -> str:

        if score >= 85:
            return "Priority 1"

        if score >= 65:
            return "Priority 2"

        if score >= 40:
            return "Priority 3"

        return "Priority 4"

    # ----------------------------------------------------------------
    # CHUNK ALIGNMENT
    # ----------------------------------------------------------------

    def scan_chunk(
        self,
        chunk: GenomeChunk
    ) -> List[OffTargetResult]:

        chunk_results = []

        orientations = [
            ("+", chunk.sequence),
            ("-", reverse_complement(chunk.sequence))
        ]

        for strand, oriented_sequence in orientations:

            maximum_starts = min(
                chunk.valid_starts,
                len(oriented_sequence) - WINDOW_LENGTH + 1
            )

            for index in range(maximum_starts):

                self.statistics.total_windows += 1

                target = oriented_sequence[
                    index:index + GUIDE_LENGTH
                ]

                pam = oriented_sequence[
                    index + GUIDE_LENGTH:
                    index + WINDOW_LENGTH
                ]

                # Reject ambiguous target and PAM sequences.
                if not (
                    all(base in VALID_BASES for base in target)
                    and all(base in VALID_BASES for base in pam)
                ):
                    continue

                if not self.is_valid_pam(pam):
                    continue

                self.statistics.valid_pam_sites += 1

                mismatch_count, mismatch_positions = (
                    self.calculate_mismatches(target)
                )

                if mismatch_count > self.max_mismatches:
                    continue

                # Calculate reference genome coordinates.
                if strand == "+":

                    genomic_start = (
                        chunk.start_position + index
                    )

                else:

                    genomic_start = (
                        chunk.start_position
                        + len(chunk.sequence)
                        - index
                        - WINDOW_LENGTH
                    )

                genomic_end = (
                    genomic_start + WINDOW_LENGTH
                )

                seed_mismatches = (
                    self.calculate_seed_mismatches(
                        mismatch_positions
                    )
                )

                gc_content = calculate_gc_content(
                    target
                )

                score = self.calculate_candidate_score(
                    mismatch_count,
                    seed_mismatches
                )

                priority = self.classify_candidate(
                    score
                )

                result = OffTargetResult(
                    chromosome=chunk.chromosome,
                    start_position=genomic_start,
                    end_position=genomic_end,
                    strand=strand,
                    target_sequence=target,
                    pam_sequence=pam,
                    mismatch_count=mismatch_count,
                    mismatch_positions=",".join(
                        map(str, mismatch_positions)
                    ),
                    seed_mismatches=seed_mismatches,
                    gc_content=gc_content,
                    candidate_score=score,
                    priority=priority
                )

                chunk_results.append(result)

        self.statistics.candidate_sites += len(
            chunk_results
        )

        return chunk_results

    # ----------------------------------------------------------------
    # COMPLETE GENOME SCAN
    # ----------------------------------------------------------------

    def scan_genome(
        self,
        fasta_path: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE
    ) -> List[OffTargetResult]:

        parser = GenomicParser(fasta_path)

        self.statistics = ScanStatistics()

        self.results = []

        start_time = time.perf_counter()

        logger.info("=" * 65)
        logger.info("GENEWEAVER STREAMING ALIGNMENT ENGINE")
        logger.info("=" * 65)

        logger.info(
            "Version: %s",
            VERSION
        )

        logger.info(
            "Guide RNA: %s",
            self.guide
        )

        logger.info(
            "Maximum mismatches: %d",
            self.max_mismatches
        )

        logger.info(
            "Chunk size: %s bases",
            f"{chunk_size:,}"
        )

        logger.info(
            "Streaming preprocessing enabled."
        )

        logger.info(
            "Starting genomic alignment..."
        )

        for chunk in parser.parse_chunks(chunk_size):

            self.statistics.total_chunks += 1

            chunk_results = self.scan_chunk(chunk)

            self.results.extend(chunk_results)

            if (
                self.statistics.total_chunks == 1
                or self.statistics.total_chunks % 100 == 0
            ):

                logger.info(
                    "Chunks: %d | Bases read: %s | Candidates: %d",
                    self.statistics.total_chunks,
                    f"{parser.total_bases:,}",
                    self.statistics.candidate_sites
                )

        self.statistics.total_bases = parser.total_bases

        self.statistics.ambiguous_bases = (
            parser.ambiguous_bases
        )

        self.statistics.execution_time = (
            time.perf_counter() - start_time
        )

        logger.info(
            "Streaming genomic alignment completed."
        )

        self.print_statistics()

        return self.results

    # ----------------------------------------------------------------
    # STATISTICS
    # ----------------------------------------------------------------

    def print_statistics(self):

        stats = self.statistics

        print("\n")
        print("=" * 68)
        print("                 GENEWEAVER SCAN REPORT")
        print("=" * 68)

        print(f"Version                : {VERSION}")
        print(f"Guide RNA              : {self.guide}")

        print(
            f"Maximum mismatches     : {self.max_mismatches}"
        )

        print(
            f"Total genomic bases    : {stats.total_bases:,}"
        )

        print(
            f"Ambiguous bases        : {stats.ambiguous_bases:,}"
        )

        print(
            f"Total chunks processed : {stats.total_chunks:,}"
        )

        print(
            f"Total windows scanned  : {stats.total_windows:,}"
        )

        print(
            f"Valid PAM sites        : {stats.valid_pam_sites:,}"
        )

        print(
            f"Off-target candidates  : {stats.candidate_sites:,}"
        )

        print(
            f"Execution time         : {stats.execution_time:.6f} seconds"
        )

        if stats.execution_time > 0:

            throughput = (
                stats.total_bases / stats.execution_time
            )

            print(
                f"Processing throughput  : {throughput:,.2f} bases/sec"
            )

        print("=" * 68)


# ====================================================================
# 7. CSV REPORT GENERATOR
# ====================================================================

def export_results_to_csv(
    results: List[OffTargetResult],
    output_path: str
) -> None:

    output_file = Path(output_path)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "chromosome",
        "start_position",
        "end_position",
        "strand",
        "target_sequence",
        "pam_sequence",
        "mismatch_count",
        "mismatch_positions",
        "seed_mismatches",
        "gc_content",
        "candidate_score",
        "priority"
    ]

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            writer.writerow({
                "chromosome": result.chromosome,
                "start_position": result.start_position,
                "end_position": result.end_position,
                "strand": result.strand,
                "target_sequence": result.target_sequence,
                "pam_sequence": result.pam_sequence,
                "mismatch_count": result.mismatch_count,
                "mismatch_positions": result.mismatch_positions,
                "seed_mismatches": result.seed_mismatches,
                "gc_content": result.gc_content,
                "candidate_score": result.candidate_score,
                "priority": result.priority
            })

    logger.info(
        "CSV report saved: %s",
        output_file.resolve()
    )


# ====================================================================
# 8. COMMAND LINE INTERFACE
# ====================================================================

def create_argument_parser():

    parser = argparse.ArgumentParser(
        prog="GeneWeaver",
        description=(
            "Streaming CRISPR off-target alignment engine."
        )
    )

    parser.add_argument(
        "--fasta",
        required=True,
        help="Input FASTA file."
    )

    parser.add_argument(
        "--guide",
        required=True,
        help="20-base CRISPR guide sequence."
    )

    parser.add_argument(
        "--mismatches",
        type=int,
        default=DEFAULT_MAX_MISMATCHES,
        help="Maximum allowed mismatches."
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Genomic chunk size."
    )

    parser.add_argument(
        "--output",
        default="off_target_results.csv",
        help="Output CSV report."
    )

    return parser


# ====================================================================
# 9. MAIN APPLICATION
# ====================================================================

def main():

    argument_parser = create_argument_parser()

    args = argument_parser.parse_args()

    try:

        engine = CPUAlignmentEngine(
            guide_sequence=args.guide,
            max_mismatches=args.mismatches
        )

        results = engine.scan_genome(
            fasta_path=args.fasta,
            chunk_size=args.chunk_size
        )

        export_results_to_csv(
            results=results,
            output_path=args.output
        )

        print("\n")
        print("=" * 55)
        print("GENEWEAVER COMMIT 2 COMPLETED")
        print("=" * 55)

        print(
            f"Candidates detected: {len(results)}"
        )

        print(
            f"Execution time: "
            f"{engine.statistics.execution_time:.6f} seconds"
        )

        print(
            f"CSV report: {Path(args.output).resolve()}"
        )

        print("=" * 55)

    except (
        ValueError,
        FileNotFoundError,
        PermissionError,
        OSError
    ) as error:

        logger.error(
            "Execution failed: %s",
            error
        )

        sys.exit(1)


# ====================================================================
# 10. ENTRY POINT
# ====================================================================

if __name__ == "__main__":
    main()
