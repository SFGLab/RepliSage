"""LEF-fork encounter counters of the kernel and the S-phase statistics."""
import numpy as np
from src.MC.runner import run_mc
from src.MC.moves import slide, slide_ev
from src.plots.style import make_phases
from src.analysis.fork_stats import fork_summary, fork_table
from tests.conftest import make_params, make_fork_matrix

def run(**kw):
    p = make_params(N=120, steps=2400, sweep=200, mc=20, rep=True, **kw); r = run_mc(p, seed=4)[0]; return p, r, make_phases(p, r.Es.shape[0])

def test_events_only_in_s_phase_and_bounded():
    p, r, ph = run()
    Fk = r.Fk; s = ph.segments()["S"]
    assert Fk.shape == (len(r.Es), 3) and Fk.sum() > 0
    assert Fk[:s.start].sum() == 0 and Fk[s.stop:].sum() == 0               # nothing outside S phase
    assert (Fk[:, 1] + Fk[:, 2] <= Fk[:, 0]).all()                          # events never exceed encounters

def test_no_events_without_pushing_and_without_replication():
    _, r, _ = run(repfork_push=False); assert r.Fk[:, 1].sum() == 0         # no pushing -> no pushed legs
    r2 = run_mc(make_params(N=120, steps=600, sweep=100, mc=20), seed=1)[0]; assert r2.Fk.sum() == 0

def test_summary_and_table():
    p, r, ph = run(); s = fork_summary(r, p, ph)
    assert s["enc"] >= s["push"] + s["rem"] and 0 <= s["push_rate"] <= 1 and 0 <= s["rem_rate"] <= 1
    for k in ("enc", "push", "rem"):
        c = s["cum"][k]; assert np.all(np.diff(c[np.isfinite(c)]) >= -1e-12) and (np.isnan(c).all() or abs(c[-1] - 1) < 1e-9)
    assert sum(s["bins"]["push"]) == s["push"] and fork_table(r, p, ph).key == "fork_events"

def test_slide_ev_matches_slide():
    f = make_fork_matrix(60, 40); rng = np.random.default_rng(0)
    for _ in range(200):
        m = int(rng.integers(0, 50)); n = m + int(rng.integers(1, 9)); t = int(rng.integers(40))
        a = slide(m, n, 60, f, t, False, True); b = slide_ev(m, n, 60, f, t, False, True)   # rw=False: deterministic
        assert a == b[:2] and 0 <= b[2] <= 2
