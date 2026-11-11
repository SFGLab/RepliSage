"""Hi-C validation: reader helpers, metrics and the pipeline with a mocked Hi-C map."""
import numpy as np
import pytest
from src.preproc.hic import Geom, rebin, candidate_resolutions, load_hic_region, have_hicstraw
from src.analysis.hic_compare import compare_hic, validate_hic, hic_tables, model_maps
from src.MC.runner import run_mc
from src.plots.style import make_phases

G = Geom("chr1", 0, 100_000, 10)       # 10 kb beads

def test_rebin_finer_and_coarser():
    fine = np.ones((20, 20)); out, cov = rebin(fine, 5_000, 0, G); assert out.shape == (10, 10) and np.allclose(out, 4) and cov.min() == 1
    coarse = np.ones((5, 5)) * 8; out, _ = rebin(coarse, 20_000, 0, G); assert np.allclose(out, 2)           # counts split by (bin/bead)^2

def test_candidate_resolutions_prefers_close_and_caps():
    assert candidate_resolutions([5000, 10000, 25000], 10_000, 100_000)[0] == 10_000
    assert candidate_resolutions([1000], 10_000, 100_000_000) == []                    # would need too many bins
    assert candidate_resolutions([5000, 10000], 10_000, 100_000, wanted=5000)[0] == 5000

@pytest.mark.skipif(have_hicstraw(), reason="hic-straw installed")
def test_missing_hicstraw_message():
    with pytest.raises(ImportError, match="hic-straw"): load_hic_region("x.hic", G)

def test_compare_self_is_perfect_and_noise_is_worse():
    n = 60; s = np.abs(np.subtract.outer(np.arange(n), np.arange(n))); rng = np.random.default_rng(0)
    m = 1 / (1 + s) + .3 * (np.add.outer(np.arange(n) // 15 % 2, np.arange(n) // 15 % 2) == 1)
    a = compare_hic(m, m); assert a["r_log"] > .999 and a["f1"] == 1.0
    noisy = m * np.exp(rng.normal(0, .8, m.shape)); noisy = (noisy + noisy.T) / 2; b = compare_hic(noisy, m)
    assert b["r_log"] < a["r_log"] and not np.isnan(b["r_pc"])

def test_validate_hic_with_mock_map(params, tmp_path):
    from types import SimpleNamespace
    p = params(rep=True, potts=True); r = run_mc(p, seed=2)[0]; ph = make_phases(p, r.Es.shape[0])
    hic = model_maps(r, p, ph, slice(ph.g1, ph.n))[0] * np.exp(np.random.default_rng(1).normal(0, .2, (p.N_beads,) * 2)); hic = (hic + hic.T) / 2; hic[5, :] = hic[:, 5] = np.nan
    ctx = SimpleNamespace(region=[0, p.N_beads * 10_000], n_beads=p.N_beads)
    V = validate_hic(ctx, p, r, ph, hic, str(tmp_path)); T = hic_tables(V, dict(file="m.hic", resolution=10000, norm="KR", coverage=.99))
    assert V["metrics"]["MC loop map"]["r_log"] > .8 and set(V["phases"]) == {"G1", "S", "G2/M"} and [t.key for t in T] == ["hic_validation", "hic_phases"]
    from src.plots.hic_validation import fig_hic
    from src.plots.style import make_geo; import os
    assert os.path.exists(fig_hic(V, dict(), SimpleNamespace(), str(tmp_path)))
