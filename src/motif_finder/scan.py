"""Orientation of CTCF sites from motif hits (PWM scan of both strands, as in LoopSage)."""
import numpy as np
from .fasta import Fasta
from .pssm import load_pfm
from ..logger import log

def anchor_call(pssm, seq, cutoff):
    """(n hits, best score, P(reverse)) of one window; P(reverse) = reverse-strand share of the hits weighted by 2^score, -1 = no motif."""
    _, strand, score = pssm.scan(seq, cutoff)
    if len(score) == 0: return 0, np.nan, -1.0
    w = 2.0 ** (score - score.max())
    return len(score), float(score.max()), float(w[strand < 0].sum() / w.sum())

def motif_orientation(chrom, starts, ends, genome, pfm=None, pvalue=1e-5, pseudocount=0.8, pad=0):
    """P(reverse) for each window [start, end) (bp) of one chromosome; -1 where no motif (or no sequence) is found.

    genome: FASTA path; pfm: PFM file (default JASPAR MA0139.1 CTCF); pad: bp added on both sides of every window.
    """
    pssm = load_pfm(pfm, pseudocount); cutoff = pssm.cutoff(pvalue); fa = Fasta(genome)
    try:
        if not fa.has(chrom): raise ValueError(f"{chrom} not found in {genome} (contigs: {', '.join(list(fa.index)[:5])}...)")
        cache = {}
        for s, e in zip(np.asarray(starts).astype(int), np.asarray(ends).astype(int)):       # every distinct window once
            if (s, e) not in cache:
                seq = fa.fetch(chrom, s - pad, e + pad)
                cache[(s, e)] = anchor_call(pssm, seq, cutoff) if len(seq) >= pssm.length else (0, np.nan, -1.0)
    finally:
        fa.close()
    p = np.array([cache[(s, e)][2] for s, e in zip(np.asarray(starts).astype(int), np.asarray(ends).astype(int))])
    log.info(f"Motif finder ({pssm.name}, score >= {cutoff:.1f}, p <= {pvalue:g}): motif at {int((p >= 0).sum())} of {len(p)} windows.")
    return p
