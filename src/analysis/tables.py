"""Statistics tables (shown with -v and written to the report and to csv)."""
import csv
from dataclasses import dataclass, field
import numpy as np
from .mc_stats import diagnose, diag_segment, ess
from .lef_stats import lef_series, coverage
from .compartments import s_phase_states, transition_matrix, switch_summary, state_fractions
from ..MC.results import summary_rows

@dataclass
class Table:
    key: str
    title: str
    header: tuple
    rows: list
    note: str = ""

def g(x, d=3):
    """Compact number format."""
    if isinstance(x, (int, np.integer)): return str(x)
    if x is None or x != x: return "n/a"
    return f"{x:.{d}g}" if abs(x) < 1e5 else f"{x:.3e}"

def bead_note():
    """Unit reminder for observables measured in simulation beads."""
    from ..plots.style import UNITS
    return f"Loop lengths are in simulation beads (1 bead = {UNITS['bead_kb']:.2f} kb)." if UNITS["bead_kb"] else "Loop lengths are in simulation beads."

def has_potts(p): return p.has_rep and (p.c_potts1 != 0 or p.c_potts2 != 0)

def observables(res, p):
    """Named per-sample series used by diagnostics and figures."""
    ls = lef_series(res, p)
    obs = {"Total energy": res.Es, "Folding energy": res.Fs, "Binding energy": res.Bs,
           "Mean loop length": ls["loop"], "Bound LEFs": ls["bound"]}
    if has_potts(p): obs.update({"Potts energy": res.Es_potts, "Magnetization": res.mags})
    return obs

def overview_table(res, p, n_ctcf, kb=None):
    return Table("overview", "Run overview", ("Quantity", "Value"), [(a, b) for a, b in summary_rows(res, p, n_ctcf, kb)])

def convergence_table(obs, ph):
    seg, where = diag_segment(ph)
    d = diagnose(obs, seg)
    rows = [(r["name"], r["n"], g(r["mean"]), g(r["sd"]), g(r["ess"], 4), g(r["iat"], 3), g(r["mcse"]), g(r["geweke"], 2), g(r["rhat"], 4)) for r in d]
    note = (f"Computed {where}. ESS = effective sample size, T_ρ = integrated autocorrelation time (samples), MCSE = Monte Carlo standard error, "
            "Geweke z within ±2 and split R̂ below 1.01 indicate a stationary, well-mixed chain. " + bead_note())
    return Table("convergence", "MC convergence diagnostics", ("Observable", "n", "Mean", "SD", "ESS", "T_ρ", "MCSE", "Geweke z", "Split R̂"), rows, note), d

def phase_table(obs, ph):
    """Mean ± SD per phase and the effect size S vs G1 (Cohen d)."""
    segs = ph.segments(); rows = []
    for name, y in obs.items():
        y = np.asarray(y, float); cells = [f"{g(y[s].mean())} ± {g(y[s].std())}" if len(y[s]) > 1 else "n/a" for s in segs.values()]
        if ph.has_rep and len(y[segs['G1']]) > 3 and len(y[segs['S']]) > 3:
            a, b = y[segs["G1"]], y[segs["S"]]; sd = np.sqrt((a.var() + b.var()) / 2)
            cells.append(g((b.mean() - a.mean()) / sd, 2) if sd > 0 else "n/a")
        rows.append((name, *cells))
    head = ("Observable", *segs.keys()) + (("d (S vs G1)",) if ph.has_rep else ())
    return Table("phases", "Observables per phase (mean ± SD)", head, rows,
                 "Samples are autocorrelated, so SD describes fluctuations, not uncertainty. d = Cohen's effect size of S relative to G1. " + bead_note())

def lef_table(res, p, ph, n_ctcf, kb):
    ls = lef_series(res, p); seg = slice(ph.g1, ph.n)
    cov = coverage(res.Ms, res.Ns, p.N_beads)[:, seg]
    ll = (res.Ns - res.Ms)[:, seg]; bound = ll[ll > 0]
    rows = [("LEFs (family 1 / 2)", f"{p.N_lef} / {p.N_lef2}"), ("CTCF sites", n_ctcf), ("bound LEFs", f"{ls['bound'][seg].mean() * 100:.1f} %"),
            ("mean loop length", f"{bound.mean() * kb:.0f} kb ({bound.mean():.1f} beads)" if bound.size else "n/a"),
            ("median / 95th percentile loop length", f"{np.median(bound) * kb:.0f} / {np.percentile(bound, 95) * kb:.0f} kb" if bound.size else "n/a"),
            ("legs on a CTCF site (left / right)", f"{ls['on_L'][seg].mean() * 100:.0f} % / {ls['on_R'][seg].mean() * 100:.0f} %"),
            ("beads inside a loop", f"{(cov > 0).mean() * 100:.1f} %"), ("mean loops per bead", f"{cov.mean():.2f}")]
    return Table("lef", "Loop extrusion statistics (after burn-in)", ("Quantity", "Value"), rows)

def compartment_table(res, ph):
    a, b, (w0, w1) = s_phase_states(res.spin_traj, ph)
    M = transition_matrix(a, b); sw = switch_summary(a, b); N = len(a)
    rows = [(f"{lab} at S start → S end", " | ".join(f"{t}: {M[i, j]}" for j, t in enumerate("BNA"))) for i, lab in enumerate("BNA")]
    rows += [("switched B→A / A→B", f"{sw['B→A']} / {sw['A→B']}"), ("net flux (A gain)", sw["B→A"] - sw["A→B"]),
             ("unchanged beads", f"{(a == b).mean() * 100:.1f} %"),
             ("A fraction start → end", f"{(a == 1).mean() * 100:.1f} % → {(b == 1).mean() * 100:.1f} %"),
             ("B fraction start → end", f"{(a == -1).mean() * 100:.1f} % → {(b == -1).mean() * 100:.1f} %")]
    return Table("compartments", "A/B compartment switching across S phase", ("Quantity", "Value"), rows,
                 "A: mean spin > 0.25, B: < −0.25, N: neutral; start/end = average over 10 % of S phase.")

def rep_table(rows):
    return Table("replication", "Replication simulation", ("Quantity", "Value"), rows)

def save_csv(tables, folder):
    import os; os.makedirs(folder, exist_ok=True)
    for t in tables:
        with open(f"{folder}/{t.key}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(t.header); w.writerows(t.rows)
