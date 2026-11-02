"""Energy and temperature of the MD run (one figure, two panels)."""
import matplotlib.pyplot as plt

def plot_md_energies(out_path, time_points, potential, kinetic, temperatures):
    """Potential / kinetic energy and temperature versus simulated time."""
    from ..plots.style import apply_style, save_fig, BLUE, ORANGE, TEAL
    apply_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    a.plot(time_points, potential, color=BLUE, label="Potential"); a.plot(time_points, kinetic, color=ORANGE, label="Kinetic")
    a.set_xlabel("Time (ps)"); a.set_ylabel("Energy (kJ/mol)"); a.legend()
    b.plot(time_points, temperatures, color=TEAL); b.set_xlabel("Time (ps)"); b.set_ylabel("Temperature (K)")
    return save_fig(fig, out_path, "md_simulation", "dynamics", "md_dynamics")
