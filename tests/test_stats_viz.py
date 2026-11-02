"""MC diagnostics, compartments, CIF writer and the 3D viewer (no OpenMM needed)."""
import numpy as np
from src.analysis.mc_stats import ess, iat, split_rhat, geweke
from src.analysis.compartments import window_state, A, B, N as NEU
from src.io.structure_io import write_cif, get_coordinates_cif
from src.viz3d.data import Traj, align
from src.viz3d.html_viewer import write_html_viewer

def ar1(n, phi, seed=0):
    r = np.random.default_rng(seed); x = np.zeros(n)
    for i in range(1, n): x[i] = phi * x[i - 1] + r.normal()
    return x

def test_ess_iid_and_correlated():
    assert ess(np.random.default_rng(1).normal(size=4000)) > 3000            # ~n for independent samples
    e = ess(ar1(8000, .9)); assert 200 < e < 1200                             # theory: n(1-phi)/(1+phi) ~ 420
    assert iat(ar1(8000, .9)) > 5

def test_rhat_and_geweke():
    x = np.random.default_rng(2).normal(size=2000)
    assert split_rhat([x]) < 1.05 and abs(geweke(x)) < 3
    assert split_rhat([x, x + 5]) > 1.5                                       # chains disagree
    assert abs(geweke(np.r_[np.zeros(1000), np.ones(1000)] + x)) > 3          # drift detected

def test_window_state_deadband():
    sp = np.array([[2, 2], [-2, -2], [0, 0.1]])
    assert list(window_state(sp, slice(0, 2))) == [A, B, NEU]

def test_cif_roundtrip_two_chains(tmp_path):
    V = np.random.default_rng(3).normal(size=(20, 3)) * 10; f = str(tmp_path / "s.cif")
    write_cif(V, 2, f); txt = open(f).read()
    assert np.allclose(get_coordinates_cif(f), V, atol=1e-3)
    assert " A 1 1 ?" in txt and " B 2 1 ?" in txt and txt.count("covale") == 18   # two chains, 9 bonds each

def test_html_viewer_self_contained(tmp_path):
    F, C, N = 6, 2, 12; V = align(np.random.default_rng(4).normal(size=(F, C * N, 3)))
    tr = Traj(V, C, N, ["G1"] * 2 + ["S"] * 2 + ["G2/M"] * 2, np.arange(F) * 10, np.zeros((F, N), np.int8), np.zeros((F, N), np.uint8),
              [np.array([[1, 5]])] * F, True)
    p = write_html_viewer(tr, str(tmp_path / "v.html"), "t"); h = open(p).read()
    assert "THREE" in h and "G2/M" in h and "/*DATA*/" not in h and "src=\"http" not in h

def test_docs_links_and_examples():
    import re, glob, pathlib, configparser
    from src.config.arg_defs import build_args, ALIASES
    root = pathlib.Path(__file__).resolve().parents[1]; args = build_args()
    for md in [root / "README.md", *(root / "markdown").glob("*.md"), root / "examples" / "README.md"]:
        for target in re.findall(r"\]\(([^)#]+)\)", md.read_text()):
            if not target.startswith("http"): assert (md.parent / target).exists(), f"{md.name}: broken link {target}"
    for f in glob.glob(str(root / "examples" / "*.ini")) + [str(root / "src" / "config.ini")]:
        cp = configparser.ConfigParser(inline_comment_prefixes=(";",)); cp.read(f)
        assert all(args.has(ALIASES.get(k.upper(), k.upper())) for s in cp.sections() for k in cp[s]), f


def test_centred_and_zoom():
    """Offset removal and zoom helpers."""
    import numpy as np
    from src.plots.style import Phases, centred
    ph = Phases(100, 10, 5, 50, 80, True)
    y, pre = centred(1e10 + np.random.default_rng(0).normal(size=100), ph)
    assert pre == "Δ" and abs(np.median(y)) < 1
    y, pre = centred(np.random.default_rng(0).normal(size=100), ph)
    assert pre == ""
