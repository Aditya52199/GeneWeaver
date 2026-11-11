
"""
================================================================
GENEWEAVER - GPU-ACCELERATED CRISPR ALIGNMENT ENGINE
================================================================

COMMIT 1: CPU BASELINE ALIGNMENT ENGINE

Implemented features:
1. FASTA genomic sequence parser
2. Leading comment and whitespace handling
3. Multiple FASTA record support
4. DNA sequence validation
5. Genome chunking with overlap
6. SpCas9 NGG PAM recognition
7. Forward and reverse strand scanning
8. Mismatch detection and position tracking
9. Duplicate candidate prevention
10. CPU performance benchmarking
11. CSV report generation
12. Command-line interface

Coordinates:
Zero-based, end-exclusive.

Note:
This is an experimental off-target candidate screening
system, not a clinically validated mutation predictor.
================================================================
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


# ================================================================
# 1. CONFIGURATION
# ================================================================

APP_NAME = "GeneWeaver"
VERSION = "0.1.0"

GUIDE_LENGTH = 20
PAM_LENGTH = 3
WINDOW_LENGTH = GUIDE_LENGTH + PAM_LENGTH

DEFAULT_CHUNK_SIZE = 1_000_000
DEFAULT_MAX_MISMATCHES = 3

VALID_BASES = set("ACGT")


# ================================================================
# 2. LOGGING
# ================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(APP_NAME)


# ================================================================
# 3. DATA STRUCTURES
# ================================================================

@dataclass
class GenomeChunk:
    chromosome: str
    sequence: str
    start_position: int
    end_position: int


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
    risk_category: str


@dataclass
class ScanStatistics:
    total_bases: int = 0
    total_chunks: int = 0
    total_windows: int = 0
    valid_pam_sites: int = 0
    candidate_sites: int = 0
    execution_time: float = 0.0


# ================================================================
# 4. DNA UTILITIES
# ================================================================

def reverse_complement(sequence: str) -> str:
    """
    Return the reverse complement of a DNA sequence.
    """

    complement = str.maketrans(
        "ACGTN",
        "TGCAN"
    )

    return sequence.translate(complement)[::-1]


def validate_guide_rna(guide: str) -> str:
    """
    Validate a 20-base CRISPR guide sequence.

    U is converted to T.
    """

    guide = guide.strip().upper().replace("U", "T")

    if len(guide) != GUIDE_LENGTH:
        raise ValueError(
            f"Guide must contain exactly {GUIDE_LENGTH} bases. "
            f"Received {len(guide)} bases."
        )

    invalid = set(guide) - VALID_BASES

    if invalid:
        raise ValueError(
            f"Invalid DNA bases in guide: {invalid}"
        )

    return guide


def clean_dna_sequence(sequence: str) -> str:
    """
    Normalize DNA sequence.

    Invalid bases are converted to N.
    Their positions are preserved.
    """

    sequence = sequence.upper().replace("U", "T")

    sequence = re.sub(r"\s+", "", sequence)

    return "".join(
        base if base in VALID_BASES else "N"
        for base in sequence
    )


# ================================================================
# 5. GENOMIC FASTA PARSER
# ================================================================

class GenomicParser:
    """
    Custom FASTA parser.

    Does not depend on BioPython's FASTA parser.

    Supports:
    - Leading comments
    - Blank lines
    - Multiple chromosome records
    - UTF-8 BOM
    - Lowercase DNA sequences
    - Ambiguous DNA characters
    - Chunk generation
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

    def parse(self) -> Iterator[Tuple[str, str]]:
        """
        Read FASTA records manually.

        Each record is returned as:
            chromosome_name, DNA_sequence

        The parser avoids BioPython parser errors.
        """

        logger.info(
            "Reading FASTA file: %s",
            self.fasta_path
        )

        chromosome = None
        sequence_parts = []
        record_count = 0

        def finish_record():

            sequence = clean_dna_sequence(
                "".join(sequence_parts)
            )

            if not sequence:
                return None

            return chromosome, sequence

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

                    # Ignore blank lines.
                    if not line:
                        continue

                    # Ignore comment lines.
                    if line.startswith(("#", ";", "!")):
                        continue

                    # Process a new FASTA record.
                    if line.startswith(">"):

                        # Save the previous record.
                        if chromosome is not None:

                            result = finish_record()

                            if result is not None:

                                record_count += 1

                                logger.info(
                                    "Parsed chromosome: %s | Length: %s bases",
                                    result[0],
                                    f"{len(result[1]):,}"
                                )

                                yield result

                        header = line[1:].strip()

                        if not header:
                            raise ValueError(
                                f"Empty FASTA header at line {line_number}."
                            )

                        chromosome = header.split()[0]

                        sequence_parts = []

                    else:

                        # Ignore content before first FASTA header.
                        if chromosome is None:
                            continue

                        sequence_parts.append(
                            re.sub(r"\s+", "", line)
                        )

            # Process the final record.
            if chromosome is not None:

                result = finish_record()

                if result is not None:

                    record_count += 1

                    logger.info(
                        "Parsed chromosome: %s | Length: %s bases",
                        result[0],
                        f"{len(result[1]):,}"
                    )

                    yield result

        except UnicodeDecodeError as error:

            raise ValueError(
                "Unable to decode FASTA file. "
                "Please provide a UTF-8 encoded file."
            ) from error

        if record_count == 0:
            raise ValueError(
                "No valid FASTA records found. "
                "Check that your file contains headers beginning with >."
            )

    def generate_chunks(
        self,
        sequence: str,
        chromosome: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE
    ) -> Iterator[GenomeChunk]:
        """
        Generate overlapping genomic chunks.

        Overlap prevents missing candidate windows
        that cross chunk boundaries.
        """

        if chunk_size < WINDOW_LENGTH:
            raise ValueError(
                f"Chunk size must be at least {WINDOW_LENGTH}."
            )

        overlap = WINDOW_LENGTH - 1

        step = chunk_size - overlap

        for start in range(0, len(sequence), step):

            end = min(
                start + chunk_size,
                len(sequence)
            )

            yield GenomeChunk(
                chromosome=chromosome,
                sequence=sequence[start:end],
                start_position=start,
                end_position=end
            )


# ================================================================
# 6. CPU ALIGNMENT ENGINE
# ================================================================

class CPUAlignmentEngine:
    """
    CPU-based CRISPR off-target candidate scanner.

    Search pattern:

        20-base target + NGG PAM

    Both forward and reverse DNA strands are examined.
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

        self._seen_sites = set()

    # ------------------------------------------------------------
    # PAM VALIDATION
    # ------------------------------------------------------------

    @staticmethod
    def is_valid_pam(pam: str) -> bool:
        """
        Validate canonical SpCas9 NGG PAM.
        """

        return (
            len(pam) == PAM_LENGTH
            and all(base in VALID_BASES for base in pam)
            and pam[1:] == "GG"
        )

    # ------------------------------------------------------------
    # MISMATCH CALCULATION
    # ------------------------------------------------------------

    def calculate_mismatches(
        self,
        target: str
    ) -> Tuple[int, List[int]]:
        """
        Calculate mismatches between guide and target.

        Positions are reported using 1-based indexing.
        """

        mismatch_positions = []

        for index in range(GUIDE_LENGTH):

            if self.guide[index] != target[index]:

                mismatch_positions.append(index + 1)

        return (
            len(mismatch_positions),
            mismatch_positions
        )

    # ------------------------------------------------------------
    # CANDIDATE PRIORITY
    # ------------------------------------------------------------

    @staticmethod
    def classify_candidate(mismatches: int) -> str:
        """
        Assign computational screening categories.

        These are not experimentally validated risk scores.
        """

        if mismatches == 0:
            return "Exact Match"

        if mismatches == 1:
            return "High Priority"

        if mismatches == 2:
            return "Medium Priority"

        return "Lower Priority"

    # ------------------------------------------------------------
    # CHUNK SCANNING
    # ------------------------------------------------------------

    def scan_chunk(
        self,
        chunk: GenomeChunk
    ) -> List[OffTargetResult]:

        chunk_results = []

        sequence_length = len(chunk.sequence)

        orientations = [
            ("+", chunk.sequence),
            ("-", reverse_complement(chunk.sequence))
        ]

        for strand, oriented_sequence in orientations:

            last_start = (
                len(oriented_sequence) - WINDOW_LENGTH
            )

            for index in range(last_start + 1):

                self.statistics.total_windows += 1

                target = oriented_sequence[
                    index:index + GUIDE_LENGTH
                ]

                pam = oriented_sequence[
                    index + GUIDE_LENGTH:
                    index + WINDOW_LENGTH
                ]

                # Reject ambiguous DNA windows.
                if not (
                    all(base in VALID_BASES for base in target)
                    and all(base in VALID_BASES for base in pam)
                ):
                    continue

                # Check SpCas9 PAM.
                if not self.is_valid_pam(pam):
                    continue

                # Calculate reference coordinates.
                if strand == "+":

                    genomic_start = (
                        chunk.start_position + index
                    )

                else:

                    genomic_start = (
                        chunk.start_position
                        + sequence_length
                        - (index + WINDOW_LENGTH)
                    )

                genomic_end = (
                    genomic_start + WINDOW_LENGTH
                )

                # Prevent duplicate sites at chunk boundaries.
                unique_key = (
                    chunk.chromosome,
                    genomic_start,
                    strand
                )

                if unique_key in self._seen_sites:
                    continue

                self._seen_sites.add(unique_key)

                self.statistics.valid_pam_sites += 1

                mismatch_count, mismatch_positions = (
                    self.calculate_mismatches(target)
                )

                if mismatch_count > self.max_mismatches:
                    continue

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
                    risk_category=self.classify_candidate(
                        mismatch_count
                    )
                )

                chunk_results.append(result)

        self.statistics.candidate_sites += len(
            chunk_results
        )

        return chunk_results

    # ------------------------------------------------------------
    # COMPLETE GENOME SCANNING
    # ------------------------------------------------------------

    def scan_genome(
        self,
        fasta_path: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE
    ) -> List[OffTargetResult]:
        """
        Execute the complete CPU baseline scan.
        """

        if chunk_size < WINDOW_LENGTH:
            raise ValueError(
                f"Chunk size must be at least {WINDOW_LENGTH}."
            )

        parser = GenomicParser(fasta_path)

        # Reset previous scan state.
        self.statistics = ScanStatistics()

        self.results = []

        self._seen_sites.clear()

        start_time = time.perf_counter()

        logger.info("=" * 65)
        logger.info("GENEWEAVER CPU ALIGNMENT ENGINE")
        logger.info("=" * 65)

        logger.info("Version: %s", VERSION)

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

        logger.info("Starting genomic scan...")

        for chromosome, sequence in parser.parse():

            self.statistics.total_bases += len(sequence)

            logger.info(
                "Scanning chromosome: %s",
                chromosome
            )

            for chunk in parser.generate_chunks(
                sequence=sequence,
                chromosome=chromosome,
                chunk_size=chunk_size
            ):

                self.statistics.total_chunks += 1

                chunk_results = self.scan_chunk(
                    chunk
                )

                self.results.extend(
                    chunk_results
                )

                logger.info(
                    "Processed chunk %d | Candidates found: %d",
                    self.statistics.total_chunks,
                    self.statistics.candidate_sites
                )

        self.statistics.execution_time = (
            time.perf_counter() - start_time
        )

        logger.info("CPU baseline scan completed.")

        self.print_statistics()

        return self.results

    # ------------------------------------------------------------
    # PERFORMANCE REPORT
    # ------------------------------------------------------------

    def print_statistics(self):

        stats = self.statistics

        print("\n")
        print("=" * 65)
        print("                 GENEWEAVER SCAN REPORT")
        print("=" * 65)

        print(
            f"Guide RNA              : {self.guide}"
        )

        print(
            f"Maximum mismatches     : {self.max_mismatches}"
        )

        print(
            f"Total genomic bases    : {stats.total_bases:,}"
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

        print("=" * 65)


# ================================================================
# 7. CSV REPORT GENERATOR
# ================================================================

def export_results_to_csv(
    results: List[OffTargetResult],
    output_path: str
) -> None:
    """
    Export off-target candidates into a CSV report.
    """

    output_file = Path(output_path)

    # Create output directory if necessary.
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
        "risk_category"
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
                "risk_category": result.risk_category
            })

    logger.info(
        "CSV report saved: %s",
        output_file.resolve()
    )


# ================================================================
# 8. COMMAND LINE INTERFACE
# ================================================================

def create_argument_parser():

    parser = argparse.ArgumentParser(
        prog="GeneWeaver",
        description=(
            "CRISPR off-target candidate detection "
            "using CPU-based genomic alignment."
        )
    )

    parser.add_argument(
        "--fasta",
        required=True,
        help="Input FASTA genome file."
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
        help="Output CSV report path."
    )

    return parser


# ================================================================
# 9. MAIN APPLICATION
# ================================================================

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
        print("GENEWEAVER EXECUTION SUCCESSFUL")
        print("=" * 55)

        print(
            f"Candidates detected: {len(results)}"
        )

        print(
            f"Execution time: "
            f"{engine.statistics.execution_time:.6f} seconds"
        )

        print(
            f"Report: {Path(args.output).resolve()}"
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


# ================================================================
# 10. APPLICATION ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()
