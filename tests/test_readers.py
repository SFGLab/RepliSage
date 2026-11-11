"""Input readers (BEDPE / BED / narrowPeak) and smoothing."""
import numpy as np
import pytest
from src.preproc.readers import read_binding, detect_format, sigma_default, smooth_sparse
from src.preproc.bedpe import preprocessing, input_stats

REG, N, CH = [1_000_000, 2_000_000], 100, "chr1"     # 10 kb beads

def write(tmp_path, name, rows):
    f = tmp_path / name; f.write_text("\n".join("\t".join(map(str, r)) for r in rows) + "\n"); return str(f)

@pytest.fixture
def bedpe(tmp_path):    # two loops; anchor 1 of loop 1 is forward (p_rev = 0), anchor 2 reverse (1)
    return write(tmp_path, "l.bedpe", [(CH, 1_100_000, 1_102_000, CH, 1_400_000, 1_402_000, 2.0, 0, 1),
                                       (CH, 1_500_000, 1_502_000, CH, 1_800_000, 1_802_000, 1.0, -1, -1)])

def test_detect_format():
    assert detect_format("a.bedpe") == "bedpe" and detect_format("a.BED") == "bed" and detect_format("x.narrowPeak.gz") == "narrowpeak"
    with pytest.raises(ValueError): detect_format("a.txt")

def test_sigma_default():
    assert sigma_default(100) == 1.5 and sigma_default(3000) == 10

def test_bedpe_orientation_and_unknown_split(bedpe):
    b = read_binding(bedpe, REG, CH, N, smooth=False)
    assert b.has_orientation and b.n_sites == 4
    assert b.L[10] == 2.0 and b.R[10] == 0 and b.R[40] == 2.0 and b.L[40] == 0     # known orientation
    assert b.L[50] == b.R[50] == .5 and b.L[80] == b.R[80] == .5                  # unknown: even split

def test_smoothing_keeps_mass_and_widens(bedpe):
    raw, sm = read_binding(bedpe, REG, CH, N, smooth=False), read_binding(bedpe, REG, CH, N, smooth=True)
    assert sm.sigma == 1.5 and np.count_nonzero(sm.L) > np.count_nonzero(raw.L)
    assert np.isclose(sm.L.sum(), raw.L.sum(), rtol=.02) and sm.n_sites == raw.n_sites == 4   # counts come from the raw profile
    Ls, Rs = sm.site_masks(); assert Ls[10] and Ls[11] and not Ls[30]

def test_bed_strand_and_probability(tmp_path):
    f = write(tmp_path, "p.bed", [(CH, 1_100_000, 1_100_400, "a", 5, "+"), (CH, 1_300_000, 1_300_400, "b", 3, "-"), (CH, 1_500_000, 1_500_400, "c", 4, ".")])
    b = read_binding(f, REG, CH, N, smooth=False)
    assert b.fmt == "bed" and b.L[10] == 5 and b.R[10] == 0 and b.R[30] == 3 and b.L[50] == b.R[50] == 2

def test_narrowpeak_weight_summit_probability(tmp_path):
    f = write(tmp_path, "p.narrowPeak", [(CH, 1_100_000, 1_102_000, "a", 900, ".", 20.0, 5, 4, 1500, 1.0), (CH, 1_300_000, 1_301_000, "b", 500, ".", 10.0, 5, 4, 100, 0.0)])
    b = read_binding(f, REG, CH, N, smooth=False)
    assert b.fmt == "narrowpeak" and b.has_orientation
    assert b.R[10] == 0 and b.L[10] == 20.0 and b.R[30] == 10.0                      # summit bead; forward p=1 -> L

def test_chrom_prefix_and_region_errors(bedpe):
    assert read_binding(bedpe, REG, "1", N, smooth=False).n_sites == 4
    with pytest.raises(ValueError): read_binding(bedpe, [5_000_000, 6_000_000], CH, N)

def test_smooth_sparse_cap():
    v = np.zeros(200); v[[20, 100]] = 1; assert smooth_sparse(v, 50)[20] > .02        # sigma is capped: peaks survive

def test_preprocessing_compat_and_stats(bedpe, tmp_path):
    L, R, n = preprocessing(bedpe, REG, CH, N); assert n == 4 and (L >= 0).all()
    f = write(tmp_path, "p.bed", [(CH, 1_100_000, 1_100_400, "a", 5, "+")])
    assert dict(input_stats(f, REG, CH, N))["peaks in the region"] == 1

def test_motif_finder_orientation(tmp_path):
    from src.motif_finder import load_pfm, motif_orientation
    cons = load_pfm().consensus(); rc = cons[::-1].translate(str.maketrans("ACGT", "TGCA"))
    rng = np.random.default_rng(0); seq = "".join(rng.choice(list("ACGT"), 3000)); seq = seq[:500] + cons + seq[519:1500] + rc + seq[1519:]
    g = tmp_path / "g.fa"; g.write_text(">chr1\n" + "\n".join(seq[i:i + 60] for i in range(0, len(seq), 60)) + "\n")
    p = motif_orientation("chr1", [450, 1450, 2400], [570, 1570, 2500], str(g))
    assert p[0] == 0 and p[1] == 1 and p[2] == -1                                   # forward, reverse, none

def test_bedpe_orientation_from_motif(tmp_path, bedpe):
    from src.motif_finder import load_pfm
    cons = load_pfm().consensus(); seq = ["A"] * 2_000_000; g = tmp_path / "g.fa"
    seq = "".join(np.random.default_rng(1).choice(list("ACGT"), 2_000_000)); i = 1_400_500; seq = seq[:i] + cons + seq[i + 19:]
    g.write_text(">chr1\n" + "\n".join(seq[k:k + 80] for k in range(0, len(seq), 80)) + "\n")
    f = write(tmp_path, "n.bedpe", [(CH, 1_100_000, 1_102_000, CH, 1_400_000, 1_402_000, 2.0)])      # no orientation columns
    b = read_binding(f, REG, CH, N, smooth=False, motif=dict(genome=str(g), pvalue=1e-5, pseudocount=.8, pad=0))
    assert b.has_orientation and b.R[10] == b.L[10] == 1.0 and b.L[40] == 2.0 and b.R[40] == 0   # anchor 1: no motif -> split; anchor 2: forward

def test_anchors_in_one_bead_count_once(tmp_path):
    f = write(tmp_path, "a.bed", [(CH, 1_100_000, 1_100_400, "a", 5, "+"), (CH, 1_103_000, 1_103_400, "b", 3, "-"), (CH, 1_105_000, 1_105_400, "c", 2, "+"), (CH, 1_500_000, 1_500_400, "d", 4, "-")])
    assert read_binding(f, REG, CH, N, smooth=False).n_sites == 2          # three anchors share bead 10 -> one site; + bead 50
