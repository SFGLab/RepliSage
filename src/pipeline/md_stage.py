"""Stage: 3D structures with OpenMM (EM or MD). One chain without replication, two chains with it.

Standalone example: run after an MC stage, e.g. from a notebook with (res, p) of mc_stage.
"""
from ..MD.model import MD_MODEL
from ..MD.units import parse_quantity
from ..logger import log

def run_md_stage(ctx, p, res, out_dir, rep):
    """Build the 3D ensemble of one MC replica into out_dir/ensemble. Returns the separation sample (or None)."""
    a = ctx.args
    log.step(f"3D structures ({a.SIMULATION_TYPE}, {2 if rep is not None else 1} chain(s)) -> {out_dir}")
    md = MD_MODEL(res.Ms, res.Ns, p.N_beads, p.burnin, p.MC_step, out_dir, a.PLATFORM,
                  None if rep is None else rep.f_rep_T, p.t_rep, res.spin_traj if rep is not None else None)
    md.device = a.DEVICE
    return md.run_pipeline(a.INITIAL_STRUCTURE_TYPE, tol=a.TOLERANCE, sim_step=a.SIM_STEP, reporters=a.DCD_REPORTER,
                           mode=a.SIMULATION_TYPE, integrator_mode=a.INTEGRATOR_TYPE, p_ev=a.EV_P,
                           md_temperature=parse_quantity(a.SIM_TEMP), integrator_step=parse_quantity(a.INTEGRATOR_STEP),
                           ff_path=a.FORCEFIELD_PATH, plot_energy=a.SAVE_PLOTS)
