"""MC results: container, summary tables and saving."""
from dataclasses import dataclass
import numpy as np
from ..logger import log

@dataclass
class MCResult:
    Ms: np.ndarray
    Ns: np.ndarray
    Es: np.ndarray
    Es_potts: np.ndarray
    Fs: np.ndarray
    Bs: np.ndarray
    spin_traj: np.ndarray
    mags: np.ndarray
    acceptance_rate: float
    N_viols: np.ndarray
    S_viols: np.ndarray
    seed: int = -1
    elapsed: float = 0.0

def phase_bounds(p):
    """Indices (in recorded samples) of the G1 start, S start and S end."""
    return p.burnin // p.MC_step, p.t_rep // p.MC_step, (p.t_rep + p.rep_duration) // p.MC_step

def summary_rows(res, p, N_CTCF, kb=None):
    """Rows for the MC statistics table."""
    g1, s0, s1 = phase_bounds(p)
    ll = res.Ns - res.Ms; ln = (lambda x: f"{x * kb:.0f} kb") if kb else (lambda x: f"{x:.2f} beads")   # loop length with units
    rows = [("acceptance rate", f"{res.acceptance_rate * 100:.2f} %"), ("MC steps", p.N_steps),
            ("recorded samples", p.N_steps // p.MC_step), ("beads", p.N_beads),
            ("CTCF sites", N_CTCF), ("LEFs (family 1 / 2)", f"{p.N_lef} / {p.N_lef2}")]
    if p.has_rep:
        nz = res.S_viols[s0:s1][res.S_viols[s0:s1] != 0]
        rows += [("energy G1 / S / G2M", " / ".join(f"{x.mean():.2f}" for x in (res.Es[g1:s0], res.Es[s0:s1], res.Es[s1:]))),
                 ("Potts energy G1 / S / G2M", " / ".join(f"{x.mean():.2f}" for x in (res.Es_potts[g1:s0], res.Es_potts[s0:s1], res.Es_potts[s1:]))),
                 ("mean loop length G1 / S / G2M", " / ".join(ln(x.mean()) for x in (ll[:, g1:s0], ll[:, s0:s1], ll[:, s1:]))),
                 ("LEF-replisome violations (S)", int(res.N_viols[s0:s1].sum())),
                 ("violation strength mean/max (S)", f"{nz.mean() if nz.size else 0:.2f} / {nz.max() if nz.size else 0:.2f}")]
    else:
        rows += [("mean energy", f"{res.Es[g1:].mean():.2f}"), ("mean loop length", ln(ll[:, g1:].mean()))]
    return rows

def check_acceptance(rate):
    if rate < 0.1: log.alert(f"The acceptance rate is low ({rate:.2f}). Consider adjusting the parameters.")
    elif rate > 0.6: log.alert(f"The acceptance rate is high ({rate:.2f}). Consider adjusting the parameters.")

def save_mc(res, p, out_path):
    """Save the same .npy files as the published version."""
    g1, s0, s1 = phase_bounds(p)
    mc, en = f"{out_path}/metadata/MCMC_output", f"{out_path}/metadata/energy_factors"
    np.save(f"{mc}/Ms.npy", res.Ms); np.save(f"{mc}/Ns.npy", res.Ns)
    np.save(f"{en}/Es.npy", res.Es); np.save(f"{en}/Fs.npy", res.Fs); np.save(f"{en}/Bs.npy", res.Bs)
    np.save(f"{mc}/loop_lengths.npy", res.Ns - res.Ms); np.save(f"{en}/Es_potts.npy", res.Es_potts)
    np.save(f"{mc}/mags.npy", res.mags); np.save(f"{mc}/spins.npy", res.spin_traj)
    if p.has_rep:
        np.save(f"{mc}/N_viols.npy", res.N_viols[s0:s1]); np.save(f"{mc}/S_viols.npy", res.S_viols[s0:s1])
