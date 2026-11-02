"""Key findings: a few plain-language, rule-based checks shown at the top of report.html (heuristics, not tests)."""
import numpy as np
from ..analysis.mc_stats import ess, iat
from ..analysis.compartments import s_phase_states, switch_summary
from ..analysis.tables import has_potts
from ..plots.style import kb, UNITS

OK, NOTE, WARN = "ok", "note", "warn"

def _mean_se(v):
    """Mean and standard error that accounts for autocorrelation (SE = SD / sqrt(ESS))."""
    v = np.asarray(v, float); e = ess(v); e = len(v) if not e == e else e
    return v.mean(), v.std(ddof=1) / np.sqrt(max(e, 1)) if len(v) > 3 else np.nan

def _change(y, ph, a, b):
    """Relative change of the mean of phase b vs phase a and the number of standard errors."""
    seg = ph.segments()
    if a not in seg or b not in seg or len(y[seg[a]]) < 4 or len(y[seg[b]]) < 4: return None
    (ma, sa), (mb, sb) = _mean_se(y[seg[a]]), _mean_se(y[seg[b]])
    if ma == 0 or not np.isfinite(sa + sb): return None
    return mb / ma - 1, abs(mb - ma) / max(np.hypot(sa, sb), 1e-12), ma, mb

def _len(beads):
    """Loop length with its genomic meaning (kb), or simulation beads if the bead size is unknown."""
    return f"{float(kb(beads)):.0f} kb" if UNITS["bead_kb"] else f"{beads:.1f} simulation beads"

def sampling(diag, where):
    """Is the chain well mixed in the stationary part?"""
    good = [d for d in diag if abs(d["geweke"]) < 2 and d["rhat"] < 1.05] if diag else []
    n = len(diag); frac = len(good) / n if n else 0
    st = OK if frac >= .8 else NOTE if frac >= .5 else WARN
    txt = ("Most observables are stationary and well mixed, so averages from this run are reliable." if st == OK else
           "Some observables still drift or mix slowly. Treat their averages with care or run longer (larger N_STEPS)." if st == NOTE else
           "Most observables have not settled. Run longer or lengthen the burn-in before drawing conclusions.")
    return dict(status=st, label="Sampling quality", value=f"{len(good)} of {n}", detail=f"observables pass the Geweke and split R̂ checks ({where}). {txt}")

def acceptance(res):
    r = res.acceptance_rate
    st = OK if .1 <= r <= .6 else NOTE
    txt = "A healthy range for Metropolis sampling." if st == OK else ("Few moves are accepted: the chain explores slowly (try a higher T_MC)." if r < .1 else "Almost all moves are accepted: the energy landscape may be too flat (try a lower T_MC).")
    return dict(status=st, label="Acceptance rate", value=f"{100 * r:.0f} %", detail=f"of proposed moves were accepted. {txt}")

def autocorrelation(obs, ph):
    """S phase is expected to be more autocorrelated because forks keep driving the system."""
    seg = ph.segments(); y = np.asarray(obs["Mean loop length"], float)
    if not ph.has_rep or "G1" not in seg or "S" not in seg or len(y[seg["S"]]) < 12 or len(y[seg["G1"]]) < 12: return None
    t1, ts = iat(y[seg["G1"]]), iat(y[seg["S"]])
    return dict(status=NOTE if ts > t1 else OK, label="Memory of the chain", value=f"T_ρ {t1:.0f} → {ts:.0f}",
                detail="integrated autocorrelation time (samples) of the loop length in G1 and in S. " +
                       ("Larger in S, as expected: forks keep changing loops and states in a semi-deterministic way, so consecutive samples stay alike."
                        if ts > t1 else "Similar in both phases, so replication does not slow the sampling here."))

def loops(obs, ph):
    y = np.asarray(obs["Mean loop length"], float); c = _change(y, ph, "G1", "S") if ph.has_rep else None
    if c is None:
        mean = y[ph.g1:].mean(); return dict(status=NOTE, label="Mean loop length", value=_len(mean), detail="averaged over the run after burn-in.")
    rel, z, ma, mb = c; sig = z > 2
    txt = ("The paper reports shorter loops during S phase because forks act as barriers; this run shows the same trend." if rel < 0 and sig else
           "Loops are longer in S than in G1 here. Check the replication barrier strength (REP_COEFF) and the fork speed." if rel > 0 and sig else
           "The change is within the sampling noise, so this run shows no clear effect of replication on loops.")
    return dict(status=OK if rel < 0 and sig else NOTE, label="Loops in S phase vs G1", value=("no change" if abs(rel) < .005 else f"{100 * rel:+.0f} %"), detail=f"mean loop length {_len(ma)} → {_len(mb)}. {txt}")

def replication(rep, rows):
    if rep is None: return None
    d = {r[0]: r[1] for r in rows}
    done, rho = d.get("replicated at the end", "n/a"), d.get("timing vs input, Spearman rho", "n/a")
    try: ok = float(str(done).split()[0]) >= 99 and float(rho) > .3
    except ValueError: ok = False
    return dict(status=OK if ok else NOTE, label="Replication program", value=f"{done}", detail=f"of DNA replicated by the end of S. The simulated timing correlates with the input timing (Spearman ρ = {rho}).")

def compartments(res, ph, p):
    if not has_potts(p): return None
    a, b, _ = s_phase_states(res.spin_traj, ph); sw = switch_summary(a, b); n = len(a); net = sw["B→A"] - sw["A→B"]
    fa, fb = (a == 1).mean(), (b == 1).mean()
    txt = ("More beads switch from B to A than the reverse, in line with the replication-driven B→A switching reported in the paper." if net > 0 else
           "Switching is balanced or favours A→B, so replication does not enlarge the A compartment in this run." if net <= 0 else "")
    return dict(status=NOTE, label="A/B compartments in S", value=f"{sw['B→A']} B→A · {sw['A→B']} A→B", detail=f"beads switched during S (net {net:+d}); A-like beads {100 * fa:.0f} % → {100 * fb:.0f} %. {txt}")

def conflicts(res, ph, p):
    if not ph.has_rep: return None
    v = int(res.N_viols[ph.s0:ph.s1].sum()); n = max(1, ph.s1 - ph.s0)
    return dict(status=OK if v / n < .05 else NOTE, label="Loops straddling forks", value=f"{v}", detail=f"conflicts in {n} S-phase samples. Few conflicts mean the replication barrier is respected.")

def key_findings(ctx, p, res, rep, ph, obs, diag, where, tables):
    """List of finding dicts (status, label, value, detail); None entries are dropped."""
    T = {t.key: t for t in tables}
    out = [sampling(diag, where), acceptance(res), autocorrelation(obs, ph), loops(obs, ph),
           replication(rep, T["replication"].rows) if "replication" in T else None, compartments(res, ph, p), conflicts(res, ph, p)]
    return [f for f in out if f]
