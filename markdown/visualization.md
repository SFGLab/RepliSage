# Outputs and visualisation

Every run explains itself. This page lists what is written and how to look at it.

## report.html

A single self-contained page: images and equations are embedded (no internet needed), so you can email it. Only the 3D viewer and movie stay as separate files next to it. Sections:

1. **The model**: a short description with the equations.
2. **Run summary**: key numbers.
3. **Input**: the loop anchors and, with the CTCF occupancy figure, the loop-spanned occupancy profile (0 to 1) together with CTCF orientation, strength and orientation confidence. With `--preferential-binding` this profile sets where LEFs load.
4. **Monte Carlo diagnostics**: energies, energy budget, distributions, autocorrelation, convergence table, running means, stationarity (first vs second half), MCSE vs samples, rolling mean and variance, rank correlations, LEF binding dynamics and compartment domain sizes. Every figure in `report.html` has a short "How to read it" note. Constant energy offsets are removed (labels start with Δ) so that fluctuations stay visible.
5. **Loop extrusion**: the LEF kymograph (one colour per LEF, with the mean loop size over time below), coverage, loop lengths, LEF binding, and the *model maps*: heatmaps for G1, S and G2/M made from the Monte Carlo state alone (LEF contacts, epigenetic checkerboard and their combination; each caption states how it is calculated).
6. **Replication**: forks, replicated fraction, initiation rate, timing.
7. **Epigenetic states and compartments**: the state trajectory as a heatmap, order measures explained in plain language (average epigenetic state, synchronisation, cluster order, entropy), the B to A and A to B switching map and the switching trajectory with the replication front.
8. **3D ensemble**: links to the viewer, the inline movies, the structures per phase and the structural-metric figures, including the compaction figure (radius of gyration and hull volume per phase) and the contact maps of the three phases in one row.
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
* **Colour modes.** *Sister chromatids* (orange and blue), *Compartments* (graded epigenetic state, blue negative to red positive), *Rainbow* (position along the chain), *Replication* (red copied, grey not yet copied).
* **Markers.** Cohesin (green) and condensin (magenta) anchors joined by thin lines (the loops), purple CTCF arrows showing the site orientation, turquoise replication forks. Distant parts fade into the backdrop (depth cueing). Drag to rotate, scroll to zoom, right-drag to pan; *Auto-rotate* and *Save PNG* are in the controls.
* **Contact map.** To the right, a map averaged over a moving window of frames (set the window and the contact radius with the sliders). The top strip shows compartments, the left strip shows replicated DNA, green and magenta dots are cohesin and condensin anchors, teal dashed lines are the forks.
* **Trajectory bar.** At the bottom, the whole trajectory is coloured by cell-cycle phase (blue G1, red S, teal G2/M) with the replicated fraction as a white curve. Click or drag to move, press Play to animate with interpolation between structures. The badge on the structure always states the current phase.

**Phase indicators.** Every figure with a time axis marks the cell cycle: G1 (blue), S (red) and G2/M (teal) are shaded and labelled, with dashed lines at the start and end of S. On replication figures with time vertically, the axis is padded and the bands carry the labels.

## movie_3d_sister.mp4 and movie_3d_compartment.mp4

!!! note "Movies take time"
    The movies are written by default. Rendering takes minutes, so add `--skip-movies` (config: `skip_movies = True`) for quick runs. The interactive HTML viewer is written either way and shows the same views.

PyVista movies (mp4) rendered like a molecular graphics program: smooth shaded tubes, ambient occlusion and shadows, and depth cueing (parts far from the camera fade into the backdrop, so they look distant). With `MOVIE_COLOR = auto` two movies are written, one coloured by **sister chromatid** (orange and blue) and one by **compartment** (graded epigenetic state, blue negative to red positive). Each frame has a badge with the current cell-cycle phase and MC step, a legend, and, next to the structure, the windowed contact map and the phase bar with a moving playhead.

* Cohesin (green) and condensin (magenta) anchors are joined by thin links, the loops.
* CTCF sites are purple arrows next to the fibre. An arrow pointing along the chain is a forward-facing site (it attracts the left leg of a LEF), an arrow pointing against the chain is a reverse-facing site, exactly as the orientation enters the model.
* Replication forks are turquoise spheres.

**Contact map (heatmap).** The map shows a soft, distance-based contact signal of the current window (nearby beads count more, so patterns are not lost in a binary threshold), drawn with a high-contrast colour map and a colour bar. Markers are placed so that you can tell what causes a feature:

* loop arcs above the diagonal for each LEF (circles = cohesin, squares = condensin), so a loop or TAD can be matched to the LEF that makes it;
* CTCF ticks on the margins, tilted by orientation (forward or reverse);
* replication forks: each replicated segment (a replication eye) is a turquoise band on the diagonal between its two forks. Each fork is a diamond on the diagonal and a triangle on both margin strips pointing the way the fork moves (left fork towards smaller positions, right fork towards larger ones), so there are no long lines across the map. A fountain or stripe starting at a diamond points to a fork; one starting at a circle or square points to a LEF.

**Phase bar.** Below the structure and the map, the bar has one cell per recorded frame, coloured by cell-cycle phase, with the replicated fraction as a white curve, MC-step ticks and a playhead. The playhead is placed in the full trajectory, also when the movie shows only a subset of the frames, and the printed step is the MC step of the current moment (`tests/test_video_bar.py` checks this).

Frames between two structures are interpolated after aligning the structures to each other, so the motion is smooth. Settings (config file): `MOVIE_COLOR` (`auto`, `sister`, `compartment`, `rainbow`, `replication`), `MOVIE_INTERP` (frames between structures), `MOVIE_CMAP` (heatmap colour map: `turbo` default, `inferno`, `magma`, `viridis`, `cividis`, `rs`; the HTML viewer uses the same one). Switch off with `--no-movie`. The HTML viewer and the movies share one palette (`src/viz3d/palette.py`), so the colours always agree.

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
| movie | `--skip-movies` | `skip_movies` |
| figures | | `save_plots` |
| figure resolution | | `fig_dpi` |

## In the code

- [`src/plots/style.py`](https://github.com/SFGLab/RepliSage/blob/main/src/plots/style.py): palette, `apply_style`, `Phases`, `shade_phases`, `band`, `centred`, `zoom`, `save_fig`
- [`src/plots/mc_timeplots.py`](https://github.com/SFGLab/RepliSage/blob/main/src/plots/mc_timeplots.py): energy, distribution, autocorrelation, convergence and running-mean figures
- [`src/plots/diag_extra.py`](https://github.com/SFGLab/RepliSage/blob/main/src/plots/diag_extra.py): stationarity, MCSE, rolling, correlation, budget, LEF dynamics, domains
- [`src/plots/loops.py`](https://github.com/SFGLab/RepliSage/blob/main/src/plots/loops.py): kymographs, coverage, loop-length figures
- [`src/plots/rep_plots_pro.py`](https://github.com/SFGLab/RepliSage/blob/main/src/plots/rep_plots_pro.py): replication figures
- [`src/plots/model_maps.py`](https://github.com/SFGLab/RepliSage/blob/main/src/plots/model_maps.py): heatmaps from the MC state alone
- [`src/plots/occupancy.py`](https://github.com/SFGLab/RepliSage/blob/main/src/plots/occupancy.py): loop-spanned CTCF occupancy and orientation
- [`src/plots/potts_order_fig.py`](https://github.com/SFGLab/RepliSage/blob/main/src/plots/potts_order_fig.py): epigenetic order measures
- [`src/report/mathml.py`](https://github.com/SFGLab/RepliSage/blob/main/src/report/mathml.py): offline equation rendering for the report
- [`src/plots/states.py`](https://github.com/SFGLab/RepliSage/blob/main/src/plots/states.py): compartment figures
- [`src/pipeline/figures_stage.py`](https://github.com/SFGLab/RepliSage/blob/main/src/pipeline/figures_stage.py): `make_figures(ctx, p, res, out, rep, ph, obs, diag, where)`: registers and writes every figure
- [`src/report/builder.py`](https://github.com/SFGLab/RepliSage/blob/main/src/report/builder.py): `write_report(...)`: builds `report.html`
- [`src/report/text.py`](https://github.com/SFGLab/RepliSage/blob/main/src/report/text.py): captions (`CAP`) and the how-to-read notes (`HOW`)
- [`src/viz3d/html_viewer.py`](https://github.com/SFGLab/RepliSage/blob/main/src/viz3d/html_viewer.py): `write_html_viewer(traj, path, title, max_frames)`
- [`src/viz3d/video.py`](https://github.com/SFGLab/RepliSage/blob/main/src/viz3d/video.py): `make_movie(tr, out_base, mode, n_interp, fps, win, size, radius, gif, max_frames)`; [`src/viz3d/palette.py`](https://github.com/SFGLab/RepliSage/blob/main/src/viz3d/palette.py): colours shared by the movie and the viewer
- [`src/viz3d/data.py`](https://github.com/SFGLab/RepliSage/blob/main/src/viz3d/data.py): `load_traj`, `list_ensemble`, `align`
- [`src/io/structure_io.py`](https://github.com/SFGLab/RepliSage/blob/main/src/io/structure_io.py): `write_cif`, `write_chimerax_script`
- [`src/pipeline/viz3d_stage.py`](https://github.com/SFGLab/RepliSage/blob/main/src/pipeline/viz3d_stage.py): `run_viz3d(ctx, p, res, out_dir, rep)`

| Argument | Role |
|---|---|
| `--no-report`, `--no-viewer` | skip the report or the HTML viewer |
| `--skip-movies [true/false]` | skip the movies (they are written by default but take minutes; `--no-movie` is an old alias) |
| `REPORT`, `HTML_3D`, `SKIP_MOVIES` | the same switches in `config.ini` (`skip_movies = True` skips the movies) |
| `MOVIE_COLOR`, `MOVIE_INTERP` | colouring (`sister`, `compartment`, `rainbow`, `replication`) and interpolated frames between structures |
| `SAVE_PLOTS`, `FIG_DPI`, `VIZ_HEATS` | figure files, resolution and contact heatmaps |

Every figure is saved through `save_fig`; setting the environment variable `RS_CHECK=1` makes it report text overlaps.
