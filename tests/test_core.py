import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from geneweaver.bioinformatics.alignment import hamming, reverse_complement
from geneweaver.bioinformatics.fasta_parser import normalize_dna, read_fasta
from geneweaver.core.exceptions import FastaFormatError
from geneweaver.bioinformatics.pam_detector import pam_matches
from geneweaver.bioinformatics.sequence_chunker import chunk_sequence
from geneweaver.bioinformatics.off_target_scorer import HeuristicScorer
from geneweaver.bioinformatics.result_export import export_results
from geneweaver.engines.cpu_engine import scan_chunk
from geneweaver.core.models import GenomicChunk

class CoreTests(unittest.TestCase):
    def test_hamming_and_reverse_complement(self):
        self.assertEqual(hamming("AAAA","AATA"),(1,(3,)))
        self.assertEqual(reverse_complement("AAGTC"),"GACTT")

    def test_normalize_ambiguous(self):
        self.assertEqual(normalize_dna("acguR"),"ACGTN")

    def test_multirecord_fasta_and_stats(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/"records.fa"; path.write_text(">a description\nACGTN\n>b\nTTAA\n",encoding="ascii")
            records,stats=read_fasta(path)
            self.assertEqual([r.identifier for r in records],["a","b"])
            self.assertEqual(stats.total_bases,9)

    def test_empty_fasta_is_rejected(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/"empty.fa"; path.write_text("",encoding="ascii")
            with self.assertRaises(FastaFormatError): read_fasta(path)

    def test_iupac_pam(self):
        self.assertTrue(pam_matches("AGG","NGG"))
        self.assertFalse(pam_matches("ATA","NGG"))

    def test_scorer_ranks_seed_mismatches_lower(self):
        score=HeuristicScorer()
        self.assertGreater(score.score([2]),score.score([15]))

    def test_export_json(self):
        with TemporaryDirectory() as folder:
            path=export_results([{"chromosome":"chr1","start":0,"end":23,"strand":"+","guide":"A","candidate":"A","pam":"AGG","mismatch_count":0,"mismatch_positions":[],"sequence_similarity":1,"predicted_off_target_score":1,"engine":"cpu"}],Path(folder)/"out.json")
            self.assertTrue(path.is_file())
            self.assertEqual(json.loads(path.read_text(encoding="utf8"))[0]["mismatch_positions"],[])

    def test_chunks_own_every_window_once(self):
        seq="A"*31; chunks=list(chunk_sequence("chr",seq,25,23))
        starts=[c.start_position+i for c in chunks for i in range(c.valid_starts)]
        self.assertEqual(starts,list(range(9)))

    def test_cpu_both_strands(self):
        guide="ACGTACGTACGTACGTACGT"
        seq=guide.encode()+b"AGG"+b"A"*10+b"CCA"+guide.encode()
        chunk=GenomicChunk("chr1",0,seq,len(seq)-22,0)
        hits,_=scan_chunk(chunk,guide,0,"NGG")
        self.assertEqual({(h["start"],h["strand"]) for h in hits},{(0,"+"),(33,"-")})

if __name__=="__main__": unittest.main()
