"""LEF-fork encounter statistics of S phase: pushing and removal counts, rates and their cumulative course.

Encounter = a proposed LEF move met by a fork: the fork pushes a leg, or the LEF straddles a fork.
Pushed = accepted slide in which a fork pushed a leg back; removed = accepted rebinding of a straddling LEF.
Rates are per encounter (events / encounters), so they do not depend on the number of sweeps.
"""
import numpy as np
from .tables import Table, g

def fork_counts(res, p, ph):
    """(encounters, pushed, removed) per S-phase sample, or None without replication / fork data."""
    if not p.has_rep or getattr(res, "Fk", None) is None: return None
    s = ph.segments()["S"]; return tuple(np.asarray(res.Fk[s, k], float) for k in range(3))

def fork_summary(res, p, ph, n_bins=10):
    """Totals, rates, cumulative fractions over S phase and per-bin counts / rates; None if there is no fork data."""
    c = fork_counts(res, p, ph)
    if c is None or len(c[0]) < 2: return None
    enc, push, rem = c; n = len(enc); x = (np.arange(n) + 1) / n          # S-phase progress, 0..1
    frac = lambda v: np.cumsum(v) / v.sum() if v.sum() > 0 else np.full(n, np.nan)
    edges = np.linspace(0, n, n_bins + 1).astype(int); sums = lambda v: np.array([v[a:b].sum() for a, b in zip(edges[:-1], edges[1:])])
    be, bp, br = sums(enc), sums(push), sums(rem); rate = lambda a, b: np.divide(a, b, out=np.full(len(b), np.nan), where=b > 0)
    steps = n * p.MC_step
    return dict(x=x, enc=enc.sum(), push=push.sum(), rem=rem.sum(), push_rate=push.sum() / enc.sum() if enc.sum() else np.nan,
                rem_rate=rem.sum() / enc.sum() if enc.sum() else np.nan, per_kstep=(1e3 * push.sum() / steps, 1e3 * rem.sum() / steps),
                cum=dict(enc=frac(enc), push=frac(push), rem=frac(rem)), bins=dict(mid=(np.arange(n_bins) + .5) / n_bins, enc=be, push=bp, rem=br,
                                                                             push_rate=rate(bp, be), rem_rate=rate(br, be)))

def half_time(x, cum):
    """S-phase progress at which a cumulative fraction reaches 50 % (nan if undefined)."""
    return float(np.interp(.5, cum, x)) if np.isfinite(cum).all() else np.nan

def fork_table(res, p, ph):
    """Table of S-phase LEF-fork statistics, or None."""
    s = fork_summary(res, p, ph)
    if s is None: return None
    rows = [("LEF-fork encounters", g(int(s["enc"]))), ("legs pushed by a fork", g(int(s["push"]))), ("LEFs removed by a fork", g(int(s["rem"]))),
            ("pushing rate (per encounter)", g(s["push_rate"])), ("removal rate (per encounter)", g(s["rem_rate"])),
            ("pushes / removals per 1000 MC steps", f"{g(s['per_kstep'][0])} / {g(s['per_kstep'][1])}"),
            ("S-phase progress at 50 % of encounters / pushes / removals", " / ".join(g(half_time(s["x"], s["cum"][k])) for k in ("enc", "push", "rem")))]
    return Table("fork_events", "LEF-fork encounters in S phase", ("Quantity", "Value"), rows,
                 "Encounter = a proposed LEF move met by a fork (the fork pushes a leg, or the LEF straddles a fork). Rates = events per encounter. "
                 "S-phase progress runs from 0 (first firing) to 1 (last locus replicated); a value below 0.5 means the events come early.")
