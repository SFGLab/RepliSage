# Outputs and visualisation

Every run explains itself. This page lists what is written and how to look at it.

## report.html

A single self-contained page (images are embedded, so you can email it). Sections:

1. **The model**: a short description with the equations.
2. **Run summary**: key numbers.
3. **Input**: the loop anchors.
4. **Monte Carlo diagnostics**: energies, energy budget, distributions, autocorrelation, convergence table, running means, stationarity (first vs second half), MCSE vs samples, rolling mean and variance, rank correlations, LEF binding dynamics and compartment domain sizes. Every figure in `report.html` has a short "How to read it" note. Constant energy offsets are removed (labels start with Δ) so that fluctuations stay visible.
5. **Loop extrusion**: LEF trajectories, coverage, loop lengths, CTCF occupancy.
6. **Replication**: forks, replicated fraction, initiation rate, timing.
7. **Epigenetic states and compartments**: how spin colour propagates, B to A and A to B switching, bar plots between the start and end of S phase.
8. **3D ensemble**: links to the viewer, the movie and the structural-metric figures.
9. **Command and parameters**: the exact command, the environment, every parameter and the config file.

Every figure comes with a short text: what it shows and what to expect. A sticky navigation bar jumps between sections.

### Colour language of the figures

| Colour | Meaning |
|---|---|
| Red | A compartment, S phase |
| Blue | B compartment, G1 |
| Teal | G2/M |
| Orange and gold | Sister chromatids A and B |
| Red, black and grey shades | Individual LEFs in the kymograph |

## viewer_3d.html: interactive 3D viewer

A self-contained page (open it in any browser, no internet needed).

* **3D structure.** Rotate with the mouse, zoom with the wheel. Both sister chromatids are drawn as tubes.
* **Colour modes.** *Sister chromatids* (orange and gold), *Compartments* (red A, blue B, grey neutral), *Rainbow* (position along the chain), *Replication* (red copied, grey not yet copied).
* **Markers.** Black spheres are LEF anchors joined by red lines (the loops); teal spheres are replication forks.
* **Contact map.** To the right, a map averaged over a moving window of frames (set the window and the contact radius with the sliders). The top strip shows compartments, the left strip shows replicated DNA, black dots are the LEF anchors, teal dashed lines are the forks.
* **Trajectory bar.** At the bottom, the whole trajectory is coloured by cell-cycle phase (blue G1, red S, teal G2/M) with the replicated fraction as a white curve. Click or drag to move, press Play to animate with interpolation between structures. The badge on the structure always states the current phase.

## movie_3d.mp4 and movie_3d.gif

A PyVista movie with the same ingredients: the structure, the windowed contact map with markers, and the phase bar with a moving playhead. Frames between two structures are interpolated linearly after aligning the structures to each other, so the motion looks smooth. Settings (config file): `MOVIE_COLOR` (`auto`, `sister`, `compartment`, `rainbow`, `replication`), `MOVIE_INTERP` (frames between structures). Switch off with `--no-movie`.

## CIF files and ChimeraX

`ensemble/ensemble_<k>_<label>.cif` are standard mmCIF files:

* one chain per DNA copy (A, and B with replication),
* one C-alpha-style bead per residue with covalent bonds between neighbours,
* the B-factor holds the position along the chain.

Open the first file with the script `ensemble/view.cxc` (in ChimeraX: File, Open). It shows beads as balls connected by sticks and colours chain A orange and chain B gold. For a rainbow along the chain, use

```text
color bfactor palette rainbow target abp
```

To open the whole trajectory, `open ensemble_*.cif` loads every structure as a model; use the model slider to move through them.

## Switching parts off

| Output | Flag | Config key |
|---|---|---|
| report.html | `--no-report` | `report` |
| viewer_3d.html | `--no-viewer` | `html_3d` |
| movie | `--no-movie` | `movie` |
| figures | | `save_plots` |
| figure resolution | | `fig_dpi` |

## In the code

- [`src/plots/style.py`](https://github.com/SFGLab/RepliSage/blob/main/src/plots/style.py): palette, `apply_style`, `Phases`, `shade_phases`, `band`, `centred`, `zoom`, `save_fig`
- [`src/plots/mc_timeplots.py`](https://github.com/SFGLab/RepliSage/blob/main/src/plots/mc_timeplots.py): energy, distribution, autocorrelation, convergence and running-mean figures
- [`src/plots/diag_extra.py`](https://github.com/SFGLab/RepliSage/blob/main/src/plots/diag_extra.py): stationarity, MCSE, rolling, correlation, budget, LEF dynamics, domains
- [`src/plots/loops.py`](https://github.com/SFGLab/RepliSage/blob/main/src/plots/loops.py): kymographs, coverage, loop-length figures
- [`src/plots/rep_plots_pro.py`](https://github.com/SFGLab/RepliSage/blob/main/src/plots/rep_plots_pro.py): replication figures
- [`src/plots/states.py`](https://github.com/SFGLab/RepliSage/blob/main/src/plots/states.py): compartment figures
- [`src/pipeline/figures_stage.py`](https://github.com/SFGLab/RepliSage/blob/main/src/pipeline/figures_stage.py): `make_figures(ctx, p, res, out, rep, ph, obs, diag, where)`: registers and writes every figure
- [`src/report/builder.py`](https://github.com/SFGLab/RepliSage/blob/main/src/report/builder.py): `write_report(...)`: builds `report.html`
- [`src/report/text.py`](https://github.com/SFGLab/RepliSage/blob/main/src/report/text.py): captions (`CAP`) and the how-to-read notes (`HOW`)
- [`src/viz3d/html_viewer.py`](https://github.com/SFGLab/RepliSage/blob/main/src/viz3d/html_viewer.py): `write_html_viewer(traj, path, title, max_frames)`
- [`src/viz3d/video.py`](https://github.com/SFGLab/RepliSage/blob/main/src/viz3d/video.py): `make_movie(tr, out_base, mode, n_interp, fps, win, size, radius, gif, max_frames)`
- [`src/viz3d/data.py`](https://github.com/SFGLab/RepliSage/blob/main/src/viz3d/data.py): `load_traj`, `list_ensemble`, `align`
- [`src/io/structure_io.py`](https://github.com/SFGLab/RepliSage/blob/main/src/io/structure_io.py): `write_cif`, `write_chimerax_script`
- [`src/pipeline/viz3d_stage.py`](https://github.com/SFGLab/RepliSage/blob/main/src/pipeline/viz3d_stage.py): `run_viz3d(ctx, p, res, out_dir, rep)`

| Argument | Role |
|---|---|
| `--no-report`, `--no-viewer`, `--no-movie` | skip the report, the HTML viewer or the movie |
| `REPORT`, `HTML_3D`, `MOVIE` | the same switches in `config.ini` |
| `MOVIE_COLOR`, `MOVIE_INTERP` | colouring (`sister`, `compartment`, `rainbow`, `replication`) and interpolated frames between structures |
| `SAVE_PLOTS`, `FIG_DPI`, `VIZ_HEATS` | figure files, resolution and contact heatmaps |

Every figure is saved through `save_fig`; setting the environment variable `RS_CHECK=1` makes it report text overlaps.
