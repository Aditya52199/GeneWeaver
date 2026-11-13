
import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple


WINDOW_LENGTH = 23
GUIDE_LENGTH = 20
PAM_LENGTH = 3
OVERLAP = WINDOW_LENGTH - 1

COMPLEMENT_TABLE = bytes.maketrans(
    b"ACGTN",
    b"TGCAN"
)

VALID_BASES = set(b"ACGT")


@dataclass
class GenomicChunk:
    chromosome: str
    start_position: int
    sequence: bytes
    valid_starts: int
    chunk_index: int


class GenomicParser:
    """
    Streaming FASTA parser.

    Reads one sequence record at a time and emits bounded chunks.
    Each possible 23-base window is assigned to exactly one chunk.
    """

    def __init__(self, fasta_path: str, chunk_size: int = 100000):
        if chunk_size < WINDOW_LENGTH:
            raise ValueError(
                f"Chunk size must be at least {WINDOW_LENGTH}."
            )

        self.fasta_path = Path(fasta_path)
        self.chunk_size = chunk_size
        self.step_size = chunk_size - OVERLAP

        if not self.fasta_path.exists():
            raise FileNotFoundError(
                f"FASTA file not found: {self.fasta_path}"
            )

    @staticmethod
    def clean_sequence(line: str) -> bytes:
        """
        Normalize sequence text to uppercase DNA.
        Unsupported characters become N.
        """
        cleaned = []

        for character in line.upper():
            if character in "ACGT":
                cleaned.append(character)
            elif character == "U":
                cleaned.append("T")
            elif character in "NRYKMSWBDHV":
                cleaned.append("N")

        return "".join(cleaned).encode("ascii")

    def parse_chunks(self) -> Iterator[GenomicChunk]:
        chromosome = None
        buffer = bytearray()
        chunk_start = 0
        chunk_index = 0

        def emit_final_chunk():
            nonlocal buffer, chunk_start, chunk_index

            valid_starts = max(0, len(buffer) - OVERLAP)

            if valid_starts > 0:
                chunk = GenomicChunk(
                    chromosome=chromosome,
                    start_position=chunk_start,
                    sequence=bytes(buffer),
                    valid_starts=valid_starts,
                    chunk_index=chunk_index
                )

                chunk_index += 1
                yield chunk

            buffer.clear()

        with self.fasta_path.open(
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as fasta_file:

            for raw_line in fasta_file:
                line = raw_line.strip()

                if not line:
                    continue

                # Ignore comments and metadata lines.
                if line.startswith(("#", ";", "!")):
                    continue

                if line.startswith(">"):
                    if chromosome is not None:
                        yield from emit_final_chunk()

                    chromosome = line[1:].split()[0]
                    buffer = bytearray()
                    chunk_start = 0
                    chunk_index = 0
                    continue

                # Ignore sequence text before the first FASTA header.
                if chromosome is None:
                    continue

                buffer.extend(self.clean_sequence(line))

                while len(buffer) >= self.chunk_size:
                    chunk = GenomicChunk(
                        chromosome=chromosome,
                        start_position=chunk_start,
                        sequence=bytes(buffer[:self.chunk_size]),
                        valid_starts=self.step_size,
                        chunk_index=chunk_index
                    )

                    yield chunk

                    chunk_index += 1
                    del buffer[:self.step_size]
                    chunk_start += self.step_size

            if chromosome is not None:
                yield from emit_final_chunk()


def reverse_complement(sequence: bytes) -> bytes:
    return sequence.translate(COMPLEMENT_TABLE)[::-1]


def count_mismatches(
    target: bytes,
    guide: bytes,
    maximum_mismatches: int
) -> Tuple[int, List[int]]:
    """
    Compare a target with the guide.

    Stops immediately when the mismatch limit is exceeded.
    Returned mismatch positions are 1-based.
    """

    mismatch_count = 0
    mismatch_positions = []

    for index in range(GUIDE_LENGTH):
        if target[index] != guide[index]:
            mismatch_count += 1

            if mismatch_count > maximum_mismatches:
                return mismatch_count, []

            mismatch_positions.append(index + 1)

    return mismatch_count, mismatch_positions


def calculate_gc_content(sequence: bytes) -> float:
    valid_count = sequence.count(b"A") + sequence.count(b"C")
    valid_count += sequence.count(b"G") + sequence.count(b"T")

    if valid_count == 0:
        return 0.0

    gc_count = sequence.count(b"G") + sequence.count(b"C")

    return round((gc_count / valid_count) * 100, 2)


def calculate_candidate_score(
    mismatch_count: int,
    seed_mismatches: int
) -> float:
    """
    A simple heuristic ranking score.

    This is not a clinically validated probability of editing.
    """
    score = 100 - (mismatch_count * 12) - (seed_mismatches * 5)

    return round(max(0.0, min(100.0, score)), 2)


def classify_priority(score: float) -> str:
    if score >= 80:
        return "High"

    if score >= 55:
        return "Medium"

    return "Low"


def scan_orientation(
    oriented_sequence: bytes,
    chromosome: str,
    chunk_start: int,
    valid_starts: int,
    guide: bytes,
    maximum_mismatches: int,
    strand: str,
    chunk_index: int
) -> Tuple[List[Dict], int]:

    candidates = []

    total_starts = max(
        0,
        len(oriented_sequence) - WINDOW_LENGTH + 1
    )

    valid_starts = min(valid_starts, total_starts)

    if valid_starts <= 0:
        return candidates, 0

    # Forward-oriented indices map directly to reference positions.
    if strand == "+":
        start_indices = range(0, valid_starts)

    # Reverse-complement indices map in the opposite direction.
    else:
        first_owned_index = total_starts - valid_starts
        start_indices = range(first_owned_index, total_starts)

    valid_pam_sites = 0

    for index in start_indices:

        # SpCas9 canonical PAM: NGG.
        pam = oriented_sequence[
            index + GUIDE_LENGTH:
            index + WINDOW_LENGTH
        ]

        if len(pam) != PAM_LENGTH:
            continue

        if pam[0] not in VALID_BASES:
            continue

        if pam[1:] != b"GG":
            continue

        valid_pam_sites += 1

        target = oriented_sequence[
            index:index + GUIDE_LENGTH
        ]

        mismatch_count, mismatch_positions = count_mismatches(
            target,
            guide,
            maximum_mismatches
        )

        if mismatch_count > maximum_mismatches:
            continue

        seed_mismatches = sum(
            1 for position in mismatch_positions
            if position >= 11
        )

        # Map the oriented window back to its reference coordinate.
        if strand == "+":
            genomic_start = chunk_start + index
        else:
            genomic_start = (
                chunk_start
                + len(oriented_sequence)
                - WINDOW_LENGTH
                - index
            )

        score = calculate_candidate_score(
            mismatch_count,
            seed_mismatches
        )

        candidates.append({
            "chromosome": chromosome,
            "start_position": genomic_start,
            "end_position": genomic_start + WINDOW_LENGTH - 1,
            "strand": strand,
            "guide": guide.decode("ascii"),
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
            "chunk_index": chunk_index
        })

    return candidates, valid_pam_sites


def scan_chunk(
    chunk: GenomicChunk,
    guide: bytes,
    maximum_mismatches: int
) -> Tuple[List[Dict], Dict]:

    all_candidates = []

    forward_candidates, forward_pam_count = scan_orientation(
        oriented_sequence=chunk.sequence,
        chromosome=chunk.chromosome,
        chunk_start=chunk.start_position,
        valid_starts=chunk.valid_starts,
        guide=guide,
        maximum_mismatches=maximum_mismatches,
        strand="+",
        chunk_index=chunk.chunk_index
    )

    reverse_sequence = reverse_complement(chunk.sequence)

    reverse_candidates, reverse_pam_count = scan_orientation(
        oriented_sequence=reverse_sequence,
        chromosome=chunk.chromosome,
        chunk_start=chunk.start_position,
        valid_starts=chunk.valid_starts,
        guide=guide,
        maximum_mismatches=maximum_mismatches,
        strand="-",
        chunk_index=chunk.chunk_index
    )

    all_candidates.extend(forward_candidates)
    all_candidates.extend(reverse_candidates)

    statistics = {
        "chunk_index": chunk.chunk_index,
        "sequence_length": len(chunk.sequence),
        "valid_starts": chunk.valid_starts,
        "pam_sites": forward_pam_count + reverse_pam_count,
        "candidates": len(all_candidates)
    }

    return all_candidates, statistics


def write_csv(results: List[Dict], output_path: str):
    fieldnames = [
        "chromosome",
        "start_position",
        "end_position",
        "strand",
        "guide",
        "target",
        "pam",
        "mismatch_count",
        "mismatch_positions",
        "seed_mismatches",
        "gc_content",
        "candidate_score",
        "priority",
        "chunk_index"
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)


def run_scan(args):
    guide_text = args.guide.upper()

    if len(guide_text) != GUIDE_LENGTH:
        raise ValueError(
            "Guide RNA must contain exactly 20 DNA bases."
        )

    if any(base not in "ACGT" for base in guide_text):
        raise ValueError(
            "Guide must contain only A, C, G and T."
        )

    if not 0 <= args.mismatches <= GUIDE_LENGTH:
        raise ValueError(
            "Mismatch limit must be between 0 and 20."
        )

    guide = guide_text.encode("ascii")

    parser = GenomicParser(
        fasta_path=args.fasta,
        chunk_size=args.chunk_size
    )

    results = []

    total_chunks = 0
    total_pam_sites = 0
    total_candidates = 0

    start_time = time.perf_counter()

    print("\nGeneWeaver - Optimized CPU Alignment")
    print("------------------------------------")
    print(f"FASTA file       : {args.fasta}")
    print(f"Guide sequence   : {guide_text}")
    print(f"Maximum mismatch : {args.mismatches}")
    print(f"Chunk size       : {args.chunk_size:,}")
    print("------------------------------------\n")

    for chunk in parser.parse_chunks():

        candidates, statistics = scan_chunk(
            chunk,
            guide,
            args.mismatches
        )

        results.extend(candidates)

        total_chunks += 1
        total_pam_sites += statistics["pam_sites"]
        total_candidates += statistics["candidates"]

        print(
            f"Chunk {total_chunks:>5} | "
            f"Chromosome: {chunk.chromosome:<15} | "
            f"PAM sites: {statistics['pam_sites']:>6} | "
            f"Candidates: {statistics['candidates']:>5}"
        )

    elapsed_time = time.perf_counter() - start_time

    results.sort(
        key=lambda item: (
            -item["candidate_score"],
            item["chromosome"],
            item["start_position"]
        )
    )

    write_csv(results, args.output)

    print("\nScan completed")
    print("------------------------------------")
    print(f"Chunks processed : {total_chunks}")
    print(f"PAM sites found  : {total_pam_sites}")
    print(f"Candidates found : {total_candidates}")
    print(f"Execution time   : {elapsed_time:.4f} seconds")
    print(f"Results saved to : {args.output}")
    print("------------------------------------")


def main():
    argument_parser = argparse.ArgumentParser(
        description="GeneWeaver CPU CRISPR alignment engine"
    )

    argument_parser.add_argument(
        "--fasta",
        required=True,
        help="Input FASTA genome file"
    )

    argument_parser.add_argument(
        "--guide",
        required=True,
        help="20-base DNA guide sequence"
    )

    argument_parser.add_argument(
        "--mismatches",
        type=int,
        default=3,
        help="Maximum allowed mismatches"
    )

    argument_parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="FASTA chunk size"
    )

    argument_parser.add_argument(
        "--output",
        default="geneweaver_results.csv",
        help="Output CSV file"
    )

    args = argument_parser.parse_args()
    run_scan(args)


if __name__ == "__main__":
    main()
