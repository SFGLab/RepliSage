# Configuration file

`config.ini` is where the detailed settings live: energies, temperatures, force-field and plotting options. The command line carries only the everyday choices (see [cli.md](cli.md)).

```bash
replisage -c my_config.ini                 # use your file
replisage -c my_config.ini --seed 7        # flags override the file
```

## How to write a `config.ini`

A config file is a short text file of `key = value` lines. You never need to write all parameters: **list only what you want to change**, everything else keeps its default.

**Step 1. Start small.** Create `my_config.ini` and write the two things RepliSage always needs, the loops and the region:

```ini
[input]
bedpe_path   = loops.bedpe        ; loops with CTCF anchors (required)
chrom        = chr14
region_start = 80835000           ; base pairs
region_end   = 98674700
```

Without a `bedpe_path` RepliSage stops, because without loops it does not know where the CTCF barriers are. If you leave out the region, the **whole chromosome** is modelled.

**Step 2. Decide what to simulate.** What runs depends on the inputs you give:

| You want | Add to the file | What runs |
|---|---|---|
| Loop extrusion only | `rept_path = none` | loop extrusion; no replication, no S or G2/M phase |
| Replication with the built-in data | `use_sc_rept = True` | + replication from the single-cell GM12878 timing, and A/B compartments |
| Replication with your own data | `rept_path = my_timing.bigWig` (or `.txt`) | the same, driven by your Repli-Seq signal |
| 3D structures | `simulation_type = MD` (default), `EM`, or `none` to switch off | a 3D model built with OpenMM; MD gives the replication trajectory |

If you give no timing at all and run in a terminal, RepliSage asks whether to use the built-in timing or to continue with loop extrusion only. `ask_rt = False` switches that question off, which is what you want for scripts and clusters.

**Step 3. Tune only when needed.** Typical first changes are the run length and the temperature in `[mc]`:

```ini
[mc]
n_steps = 150000      ; Monte Carlo steps
mc_step = 200         ; record every 200th step
burnin  = 5000        ; steps discarded at the start
t_mc    = 1.5         ; Monte Carlo temperature
```

**Step 4. Run and check.** Every run saves the values it actually used as `config_used.ini` in its folder. Open it to confirm that your changes were picked up, and use it to repeat the run exactly.

```bash
replisage -c my_config.ini -v
```

### A complete example

```ini
; my_config.ini: loops + replication + 3D structures for a 17.8 Mb region
[general]
out_path = results
run_name = chr14_demo
n_threads = 4
seed = 7

[input]
bedpe_path   = loops.bedpe
rept_path    = GM12878_timing.bigWig    ; Repli-Seq, always log-normalised
chrom        = chr14
region_start = 80835000
region_end   = 98674700
n_beads      = 1000                     ; empty = 7 kb per bead (max 10,000 beads)

[replication]
rep_start_time    = 50000               ; MC step where S phase starts
rep_time_duration = 50000               ; length of S phase in MC steps

[mc]
n_steps = 150000
t_mc    = 1.5

[md]
simulation_type = MD
platform = CUDA
```

### Common mistakes

* **A typo in a key.** Unknown keys are ignored with a warning (`-vv` shows it), so check `config_used.ini` if a change seems to have no effect.
* **Comments.** `;` starts a comment, also after a value. Do not put `#` or quotes around values.
* **Empty is not zero.** An empty value means "not set", so the default is used. Write `0` if you mean zero.
* **Coordinates in kb or Mb.** Region coordinates are always in base pairs.
* **Region too small for the number of beads.** The bead size is `(region_end - region_start) / n_beads`; keep it at a few kb or more.
* **Replication time too short.** `rep_time_duration` must be larger than `rep_sim_t`, and the S phase must fit inside `n_steps`.

The repository also ships a fully annotated template, `src/config.ini`, and ready-made variants in the [examples folder](examples.md), and short runnable ones in `demo/`.

## Rules

* **Everything has a key.** Each command-line flag is a shortcut for a key, and the file has many more. `src/config.ini` is a full commented template. With a well-defined file `replisage -c my_config.ini` is all you need.
* **Checks.** `N_LEF + N_LEF2` at or above half the beads is an error. Bead size outside 200 bp - 50 kb is an error, outside 1 - 10 kb a warning.
* `rept_path` accepts a `.bigWig/.txt` file, a single-cell `.parquet`, `builtin` (shipped GM12878 data) or `none` (loop extrusion only).

* The format is a plain ini file. Section names are only for readability: `[mc]`, `[md]` and so on can be named anything, and a key may sit in any section.
* Keys are not case sensitive. `N_LEF`, `n_lef` and `N_lef` are the same.
* `;` starts a comment, also after a value.
* An empty value means "not set" and the default is used.
* Unknown keys are ignored with a warning, so typos do not stop a run but show up in the log.
* Booleans accept `True/False`, `yes/no`, `1/0`.
* Precedence: defaults, then the file, then the command-line flags.
* Every run saves the resolved parameters as `config_used.ini` in its folder, so you can repeat any run with `-c` on that file.

## All parameters

One table per section of the file. *Flag* is the command-line option that sets the same thing, if there is one. *Type* tells how to write the value: `integer`, `number` (decimals allowed), `text` (a path or a name) or `True/False`. These tables are generated from `src/config/arg_defs.py` by `docs/scripts/gen_config_tables.py`, so they always match the code.

### General (`[general]`)

| Key | Flag | Type | Default | Meaning |
|---|---|---|---|---|
| `platform` | `--platform` | text | auto | OpenMM platform: auto (default) tries CUDA, then OpenCL, then CPU and falls back on any error; or force CUDA, OpenCL or CPU. |
| `device` |  | text | empty | Device index for CUDA or OpenCL (count from 0). |
| `out_path` | `--out-path` | text | results | Root folder of results. Each run gets its own sub-folder inside. |
| `run_name` | `--run-name` | text | empty | Exact name of the run sub-folder. Empty = automatic name (chrom + region + time). |
| `run_tag` |  | text | empty | Free label appended to the automatic run name. |
| `n_threads` | `--threads` | integer | 1 | Number of CPU threads used to run independent MC replicas in parallel. |
| `seed` | `--seed` | integer | empty | Random seed of the Monte Carlo (empty = random). |
| `verbose` |  | integer | 0 | Verbosity 0-3 (same as -v, -vv, -vvv): 0 stages, progress bars and results; 1 + input / output tables; 2 + detailed info; 3 everything. |

### Input (`[input]`)

| Key | Flag | Type | Default | Meaning |
|---|---|---|---|---|
| `bedpe_path` | `--bedpe-file` | text | empty | A .bedpe file path with loops (CTCF anchors). It is required: without it the model does not know where the CTCF barriers are. Optional columns 8-9 hold CTCF orientation. |
| `rept_path` | `--rt-file` | text | empty | The replication timing dataset (.txt, .bw, .bigwig). If empty, no replication is modelled (unless USE_SC_REPT). |
| `sc_rept_path` | `--rt-file` | text | built-in file | The single cell replication timing dataset (.parquet). |
| `use_sc_rept` | `--rt-file builtin` | True/False | False | Use the built-in single-cell replication timing when REPT_PATH is empty. |
| `ask_rt` | `--rt-file none` | True/False | True | Ask interactively what to do when no replication timing is given (only in a terminal). |
| `chrom` | `--region` | text | empty | Chromosome of the modelling region of interest. |
| `region_start` | `--region` | integer | empty | Starting region coordinate. Empty = whole chromosome. |
| `region_end` | `--region` | integer | empty | Ending region coordinate. Empty = whole chromosome. |
| `n_beads` | `--n-beads` | integer | empty | Number of Simulation Beads. Empty = region length / 7 kb (about 10 beads per CTCF site; max 10,000 beads). Bead sizes outside 200 bp - 50 kb are an error. |

### Pre-processing (`[preprocessing]`)

| Key | Flag | Type | Default | Meaning |
|---|---|---|---|---|
| `rep_sim_t` |  | integer | 1000 | Number of time points of the replication simulation. |
| `bigwig_bin_size` |  | integer | 1000 | Bin size (bp) used to average a Repli-Seq bigWig. |
| `bigwig_sigma` |  | number | 200.0 | Gaussian smoothing (in bins) of a Repli-Seq bigWig. |
| `bigwig_log_norm` |  | True/False | True | [deprecated] Repli-Seq signals are always log-normalised; this value is ignored. |

### Replication (`[replication]`)

| Key | Flag | Type | Default | Meaning |
|---|---|---|---|---|
| `rep_with_stress` | `--with-stress` | True/False | False | True sets REP_T_STD_FACTOR, REP_SPEED_SCALE and REP_INIT_RATE_SCALE to replication-stress values. |
| `rep_t_std_factor` |  | number | 0.1 | Factor multiplying the standard deviation of the replication timing curve. |
| `rep_speed_scale` |  | number | 10 | A scale that quantifies the speed of the replication forks. |
| `rep_init_rate_scale` |  | number | 1.0 | Multiplier of the experimentally estimated initiation rate. |
| `rep_start_time` | `--rep-start` | integer | 50000 | MC step at which replication starts (G1 = the steps before it). |
| `rep_time_duration` | `--rep-duration` | integer | 50000 | Duration of replication (in MC steps); G2/M = the steps after S phase. |

### Monte Carlo (`[mc]`)

| Key | Flag | Type | Default | Meaning |
|---|---|---|---|---|
| `lef_rw` |  | True/False | True | Random walk for cohesins instead of unidirectional sliding. |
| `cohesin_blocks_condensin` |  | True/False | False | Allow cohesins to block condensins during simulation. |
| `random_init_spins` |  | True/False | True | Use a random initial distribution of epigenetic states. |
| `p_rew` |  | number | 0.5 | Probability of proposing a rewiring move (LEF move). |
| `rep_fork_push` |  | True/False | True | Hard-coded replication fork pushing in the MC moves. |
| `rep_fork_epigenetic_organizer` |  | True/False | True | Model replication forks as epigenetic organizers. |
| `n_steps` | `--n-steps` | integer | 150000 | Total number of Monte Carlo steps (default: G1 50000 + S 50000 + G2/M 50000). |
| `n_sweep` |  | integer | 1000 | Number of node or link state changes proposed per simulation step. |
| `n_lef` |  | integer | empty | Number of loop extrusion factors (LEFs). Empty = half the number of CTCF sites (changed from twice in earlier versions). |
| `n_lef2` |  | integer | 0 | Number of second-family LEFs (different speed). |
| `mc_step` | `--mc-step` | integer | 200 | Sampling frequency of MC steps (avoids autocorrelation). |
| `burnin` | `--burnin` | integer | 5000 | Burn-in period (steps before equilibrium). |
| `t_mc` |  | number | 1.5 | Temperature of the stochastic MCMC model. |
| `folding_coeff` |  | number | 1.0 | Folding coefficient for the first family of LEFs. |
| `folding_coeff2` |  | number | 0.0 | Folding coefficient for the second family of LEFs. |
| `rep_coeff` |  | number | 0.01 | Replication-barrier penalty (energy scale = coeff * 1e8; was 1.0 = 1e8 in the published v1.0). |
| `epi_interact_coeff` |  | number | 1.0 | Interaction coefficient of the epigenetic state model. |
| `epi_field_coeff` |  | number | 1.0 | Field coefficient of the epigenetic state model. |
| `cross_coeff` |  | number | 1.0 | Coefficient for LEF crossing. |
| `bind_coeff` |  | number | 1.0 | Coefficient for CTCF binding. |
| `preferential_binding` | `--preferential-binding` | True/False | False | LEFs load preferentially where loops span (CTCF occupancy), and rarely in loop-naked areas. Also --preferential-binding. |
| `pref_bind_floor` |  | number | 0.05 | Loading weight of loop-naked beads relative to the most occupied bead (0-1) when PREFERENTIAL_BINDING is on. |
| `pref_bind_power` |  | number | 1.0 | Exponent applied to the occupancy before loading (1 = proportional, >1 = stronger preference). |
| `n_replicas` | `--n-replicas` | integer | 1 | Independent MC chains; each is processed into its own replica_XX sub-folder. |

### Molecular dynamics (`[md]`)

| Key | Flag | Type | Default | Meaning |
|---|---|---|---|---|
| `initial_structure_type` |  | text | rw | rw, confined_rw, self_avoiding_rw, helix, circle, spiral, sphere, hilbert. |
| `simulation_type` | `--run-md / --md-mode` | text | MD | 3D engine: MD (one minimization + dynamics, the default: gives the replication trajectory) or EM (many minimizations). none/empty = stop after the Monte Carlo. |
| `dcd_reporter` |  | True/False | False | Save a .dcd video of the 3D structure. |
| `integrator_type` |  | text | langevin | langevin or brownian. |
| `integrator_step` | `--md-step-size` | text | 100 femtosecond | Integrator time step (e.g. '100 femtosecond'). |
| `forcefield_path` |  | text | built-in file | Path to XML file with the forcefield. |
| `ev_p` |  | number | 0.01 | Probability that excluded volume of a bead is randomly disabled. |
| `tolerance` |  | number | 1.0 | Stopping tolerance of the energy minimization. |
| `sim_temp` |  | text | 310 kelvin | Temperature of the 3D simulation (e.g. '310 kelvin'). |
| `sim_step` | `--md-steps` | integer | 1000 | MD steps performed each time the loop forces change. |

### Output (`[output]`)

| Key | Flag | Type | Default | Meaning |
|---|---|---|---|---|
| `save_plots` |  | True/False | True | Save diagnostic plots. |
| `save_mdt` |  | True/False | True | Save metadata of the stochastic simulation. |
| `report` | `--no-report` | True/False | True | Write report.html (model summary, figures, tables, command and all parameters). |
| `html_3d` | `--no-viewer` | True/False | True | Write an interactive 3D viewer (viewer_3d.html) of the ensemble. |
| `skip_movies` | `--skip-movies` | True/False | False | Skip the PyVista movies (mp4) of the ensemble. By default they are written; they take time, so use `--skip-movies` for quick runs. |
| `movie_color` |  | text | auto | Movie colouring: auto (sister + compartment movies), sister, compartment, rainbow or replication. |
| `movie_cmap` |  | text | turbo | Heatmap colour map of movie and viewer: turbo, inferno, magma, viridis, cividis or rs. |
| `movie_interp` |  | integer | 5 | Interpolated frames between two structures in the movie. |
| `fig_dpi` |  | integer | 200 | Resolution (dpi) of the saved figures and of the report. |
| `viz_heats` |  | True/False | True | Save the averaged heatmaps of the 3D ensemble. |


## Changing the physics in detail

| You want to change | Go to |
|---|---|
| Strength of loops, CTCF binding, crossing, replication barrier, epigenetic field and coupling, temperature | `[mc]`, see [mc_model.md](mc_model.md) |
| Fork speed, origin spread, replication stress, S-phase timing | `[replication]`, see [replication_model.md](replication_model.md) |
| Integrator, 3D temperature, excluded volume, starting shape, force field | `[md]`, see [md_model.md](md_model.md) |
| Figure resolution, movie and viewer settings | `[output]`, see [visualization.md](visualization.md) |

To change force-field terms beyond these parameters (bond strengths, compartment depths), edit the XML in `src/forcefields/` or the force definitions in `src/MD/forces.py`.

## In the code

| Function or class | What it does |
|---|---|
| `build_args()` in `config/arg_defs.py` | the table of every parameter: section, name, help, type and default |
| `Args`, `Arg`, `cast` in `config/arg_types.py` | `args.N_BEADS` style access and casting of strings to python types |
| `read_ini(args, path)` | reads a config file; an unknown key is ignored with a warning |
| `apply_stress(args)` | replication stress values used by `--with-stress` |
| `write_config(args, path)` | writes `config_used.ini` into the run folder |

To add a parameter, add one line `A(section, NAME, help, type, default)` in `src/config/arg_defs.py`; it then works in the ini file and in `args.NAME`.
