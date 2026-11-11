"""Energy terms, budget table and landscapes."""
import numpy as np
from src.MC.runner import run_mc
from src.analysis.free_energy import energy_terms, hard_terms, budget_table, landscape
from src.plots.style import make_phases

def test_terms_add_up_and_table(params):
    p = params(rep=True, potts=True, n_lef2=4); r = run_mc(p, seed=3)[0]; T = energy_terms(r, p)
    assert np.allclose(sum(v for k, v in T.items() if k != "Total"), T["Total"])
    assert set(hard_terms(r, p)) == {"Crossing penalties", "Fork penalties"}
    t = budget_table(r, p, make_phases(p, r.Es.shape[0])); assert t.header[0] == "Energy term" and len(t.rows) == 6

def test_loop_extrusion_only_total_matches_running_energy(params):
    p = params(); r = run_mc(p, seed=1)[0]; assert set(energy_terms(r, p)) == {"Folding", "Binding", "Total"}
    assert np.allclose(r.Es, r.Et) and np.allclose(r.Et, energy_terms(r, p)["Total"] + r.Ec)      # running energy is exact without replication

def test_landscape_minimum_zero():
    x = np.random.default_rng(0).normal(0, 1, 5000); c, F = landscape(x, 1.5, -3, 3)
    assert np.nanmin(F) == 0 and abs(c[np.nanargmin(F)]) < .6
