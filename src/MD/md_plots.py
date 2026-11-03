"""Energy and temperature of the MD run (one figure, two panels, cell-cycle phases marked)."""
import matplotlib.pyplot as plt

def plot_md_energies(out_path, time_points, potential, kinetic, temperatures, labels=None):
    """Potential / kinetic energy and temperature versus simulated time; labels = BR/R/AR/LE per structure."""
    from ..plots.style import apply_style, save_fig, phase_bands, BLUE, ORANGE, TEAL
    from ..viz3d.data import PHASE_OF
    apply_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    a.plot(time_points, potential, color=BLUE, label="Potential"); a.plot(time_points, kinetic, color=ORANGE, label="Kinetic")
    a.set_xlabel("Time (ps)"); a.set_ylabel("Energy (kJ/mol)"); a.legend(loc="center right")
    b.plot(time_points, temperatures, color=TEAL); b.set_xlabel("Time (ps)"); b.set_ylabel("Temperature (K)")
    ph = [PHASE_OF.get(l, l) for l in labels or []]
    if len(ph) == len(time_points) and set(ph) != {"All"}:
        for ax in (a, b): phase_bands(ax, ph, time_points)
    return save_fig(fig, out_path, "md_simulation", "dynamics", "md_dynamics")
