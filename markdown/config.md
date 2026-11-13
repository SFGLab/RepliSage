# Configuration file

`config.ini` holds every parameter of RepliSage: energies, temperatures, force field and plotting options, besides the input paths. The command line carries only the everyday ones (see [cli.md](cli.md)).

```bash
replisage -c my_config.ini                 # use your file
replisage -c my_config.ini --seed 7        # flags override the file
```

## File format

A plain text file of `key = value` lines. List only what you want to change; everything else keeps its default.

```ini
[input]
bedpe_path   = loops.bedpe       ; see Input data
chrom        = chr14
region_start = 80835000          ; base pairs
region_end   = 98674700
```

* Section names are only for readability: a key may sit in any section.
* Keys are not case sensitive (`N_LEF` = `n_lef`).
* `;` starts a comment, also after a value. Do not use `#` or quotes.
* An empty value means "not set" (the default is used); write `0` for zero.
* Booleans accept `True/False`, `yes/no`, `1/0`.
* Unknown keys are ignored with a warning (`-vv`), so check `config_used.ini` if a change has no effect.
* Precedence: defaults, then the file, then the command-line flags.
* Every run saves the values it used as `config_used.ini` in its folder; re-run it with `-c`.
* Checks: `N_LEF + N_LEF2` at or above half the beads is an error; bead sizes outside 200 bp - 50 kb are an error, outside 1 - 10 kb a warning.

The data files themselves are described in [Input data](input_data.md); complete files are in [Examples](examples.md) and `src/config.ini`.

## All parameters

One table per section. *Flag* is the command-line option for the same setting, if there is one. *Type* is `integer`, `number`, `text` (a path or a name) or `True/False`. The tables are generated from `src/config/arg_defs.py` by `docs/scripts/gen_config_tables.py`.

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
| `bedpe_path` | `--bedpe-file` | text | empty | Input file with the CTCF sites: .bedpe (loops; optional columns 8-9 = CTCF orientation), .bed or .narrowPeak (CTCF peaks; no loops). Required: without it the model does not know where the CTCF barriers are. Also INPUT_PATH, --input-file. |
| `rept_path` | `--rt-file` | text | empty | The replication timing dataset (.txt, .bw, .bigwig). If empty, no replication is modelled (unless USE_SC_REPT). |
| `sc_rept_path` | `--rt-file` | text | built-in file | The single cell replication timing dataset (.parquet). |
| `use_sc_rept` | `--rt-file builtin` | True/False | False | Use the built-in single-cell replication timing when REPT_PATH is empty. |
| `ask_rt` | `--rt-file none` | True/False | True | Ask interactively what to do when no replication timing is given (only in a terminal). |
| `chrom` | `--region` | text | empty | Chromosome of the modelling region of interest. |
| `region_start` | `--region` | integer | empty | Starting region coordinate. Empty = whole chromosome. |
| `region_end` | `--region` | integer | empty | Ending region coordinate. Empty = whole chromosome. |
| `n_beads` | `--n-beads` | integer | empty | Number of Simulation Beads. Empty = region length / 7 kb (about 10 beads per CTCF site; max 10,000 beads). |

### Pre-processing (`[preprocessing]`)

| Key | Flag | Type | Default | Meaning |
|---|---|---|---|---|
| `smooth_input` |  | True/False | True | Smooth the CTCF binding profiles with a Gaussian (as LoopSage): sites become soft barriers instead of single beads. |
| `smooth_sigma` | `--smooth-sigma` | number | empty | Width (sigma, in beads) of the input smoothing. Empty = LoopSage default max(1.5, N_BEADS / 300). |
| `motif_genome` | `--motif-genome` | text | empty | Reference genome FASTA (uncompressed). If given, CTCF orientation is read from the motif (JASPAR MA0139.1) at every anchor or peak without orientation in the input file. |
| `motif_pfm` |  | text | empty | Own motif as a PFM file (4 rows A C G T). Empty = built-in CTCF motif. |
| `motif_pvalue` |  | number | 1e-5 | Score threshold of the motif scan: chance of a hit in random sequence. |
| `motif_pseudocount` |  | number | 0.8 | Pseudocount of the motif matrix (avoids infinite scores). |
| `motif_pad` |  | integer | 0 | bp added on both sides of every anchor or peak before scanning (the site may be a bit outside it). |
| `rep_sim_t` |  | integer | 1000 | Number of time points of the replication simulation. |
| `bigwig_bin_size` |  | integer | 1000 | Bin size (bp) used to average a Repli-Seq bigWig. |
| `bigwig_sigma` |  | number | 200.0 | Gaussian smoothing (in bins) of a Repli-Seq bigWig. |
| `bigwig_log_norm` |  | True/False | True | [deprecated] Repli-Seq signals are always log-normalised; this value is ignored. |
| `hic_file` | `--hic` | text | empty | Hi-C map (.hic) of the region: the model is validated against it (needs hic-straw). Empty = no validation. |
| `hic_norm` |  | text | KR | Hi-C normalisation: KR, SCALE, VC_SQRT, VC or NONE (the next available is used if missing). |
| `hic_resolution` |  | integer | empty | Hi-C bin size (bp) to read. Empty = the one closest to the bead size. |
| `hic_track_smooth` |  | number | 7.0 | Smoothing (beads) of the insulation and A/B tracks of the Hi-C validation. |

### Replication (`[replication]`)

| Key | Flag | Type | Default | Meaning |
|---|---|---|---|---|
| `rep_with_stress` | `--with-stress` | True/False | False | True sets REP_T_STD_FACTOR, REP_SPEED_SCALE and REP_INIT_RATE_SCALE to replication-stress values. |
| `rep_t_std_factor` |  | number | 0.1 | Factor multiplying the standard deviation of the replication timing curve. |
| `rep_speed_scale` |  | number | 10 | A scale that quantifies the speed of the replication forks. |
| `rep_init_rate_scale` |  | number | 1.0 | Multiplier of the experimentally estimated initiation rate. |
| `rep_front_smooth` |  | number | 2.0 | Smoothing (sigma, in beads) of the replication front: the replication time of neighbouring beads is averaged. 0 = sharp front. |
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
| `preferential_binding` | `--preferential-binding` | True/False | False | LEFs load preferentially where loops span (CTCF occupancy), and rarely in loop-naked areas. Also --preferential-binding. |
| `pref_bind_floor` |  | number | 0.05 | Loading weight of loop-naked beads relative to the most occupied bead (0-1) when PREFERENTIAL_BINDING is on. |
| `pref_bind_power` |  | number | 1.0 | Exponent applied to the occupancy before loading (1 = proportional, >1 = stronger preference). |
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
| `skip_movies` | `--skip-movies` | True/False | False | Skip the PyVista movies (mp4) of the ensemble. By default the movies are written (they take time: use --skip-movies to disable). |
| `movie_color` |  | text | auto | Movie colouring: auto (sister + compartment movies), sister, compartment, rainbow or replication. |
| `movie_cmap` |  | text | turbo | Colour map of the contact map in the movie and the viewer: turbo (most variable), inferno, magma, viridis, cividis or rs. |
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
