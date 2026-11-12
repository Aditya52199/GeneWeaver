
from pathlib import Path
from tempfile import TemporaryDirectory

from geneweaver import (
    GenomicParser,
    CPUAlignmentEngine,
    GenomeChunk
)


def test_streaming_parser():

    sequence = (
        "ACGT" * 300
    )

    with TemporaryDirectory() as temp:

        fasta = Path(temp) / "test.fasta"

        fasta.write_text(
            "# Test genome\n"
            ">chr1\n"
            + "\n".join(
                sequence[i:i + 60]
                for i in range(0, len(sequence), 60)
            ),
            encoding="utf-8"
        )

        parser = GenomicParser(str(fasta))

        chunks = list(
            parser.parse_chunks(chunk_size=100)
        )

        assert parser.total_bases == len(sequence)

        assert len(chunks) > 1

        assert all(
            chunk.valid_starts > 0
            for chunk in chunks
        )

        print("PASS: Streaming FASTA parser")

        print("Total bases:", parser.total_bases)

        print("Chunks:", len(chunks))


def test_chunk_boundary_alignment():

    guide = "ACGTACGTACGTACGTACGT"

    sequence = "A" * 90 + guide + "AGG" + "A" * 100

    engine = CPUAlignmentEngine(
        guide_sequence=guide,
        max_mismatches=0
    )

    chunks = []

    parser_chunk_size = 100

    overlap = 22

    step = parser_chunk_size - overlap

    for start in range(0, len(sequence), step):

        end = min(
            start + parser_chunk_size,
            len(sequence)
        )

        current = sequence[start:end]

        if len(current) < 23:
            continue

        valid_starts = min(
            step,
            len(current) - 22
        )

        if end == len(sequence):

            valid_starts = len(current) - 22

        chunks.append(
            GenomeChunk(
                chromosome="chr1",
                sequence=current,
                start_position=start,
                end_position=end,
                valid_starts=valid_starts
            )
        )

    results = []

    for chunk in chunks:
        results.extend(
            engine.scan_chunk(chunk)
        )

    matches = [
        result for result in results
        if result.start_position == 90
        and result.strand == "+"
    ]

    assert len(matches) == 1, (
        "Boundary alignment failed."
    )

    print("PASS: Chunk boundary alignment")

    print("Detected location:", matches[0].start_position)


if __name__ == "__main__":

    test_streaming_parser()

    test_chunk_boundary_alignment()

    print("\nAll Commit 2 tests passed.")
