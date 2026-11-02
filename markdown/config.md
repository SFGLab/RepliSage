# Configuration file

`config.ini` is where the detailed settings live: energies, temperatures, force-field and plotting options. The command line carries only the everyday choices (see [cli.md](cli.md)).

```bash
replisage -c my_config.ini                 # use your file
replisage -c my_config.ini --seed 7        # flags override the file
```

The repository ships a fully annotated template, `src/config.ini`. Copy it and edit what you need; ready-made variants are in the [examples folder](examples.md).

## Rules

* The format is a plain ini file. Section names are only for readability: `[mc]`, `[md]` and so on can be named anything, and a key may sit in any section.
* Keys are not case sensitive. `N_LEF`, `n_lef` and `N_lef` are the same.
* `;` starts a comment, also after a value.
* An empty value means "not set" and the default is used.
* Unknown keys are ignored with a warning, so typos do not stop a run but show up in the log.
* Booleans accept `True/False`, `yes/no`, `1/0`.
* Precedence: defaults, then the file, then the command-line flags.
* Every run saves the resolved parameters as `config_used.ini` in its folder, so you can repeat any run with `-c` on that file.

## Minimal file

```ini
[input]
bedpe_path = loops.bedpe
rept_path  = GM12878_timing.bigWig
chrom = chr14
region_start = 80835000
region_end   = 98674700

[mc]
n_steps = 100000
t_mc = 1.5

[md]
simulation_type = MD
platform = CUDA
```

## All parameters

The second column shows the command-line flag that sets the same thing, if there is one.

### General (`[general]`)

| Key | Flag | Default | Meaning |
|---|---|---|---|
| `platform` | `--platform` | CPU | OpenMM platform: CPU, CUDA or OpenCL. |
| `device` |  | empty | Device index for CUDA or OpenCL (count from 0). |
| `out_path` | `--out` | results | Root folder of results. Each run gets its own sub-folder inside. |
| `run_name` | `--name` | empty | Exact name of the run sub-folder. Empty = automatic name (chrom + region + time). |
| `run_tag` |  | empty | Free label appended to the automatic run name. |
| `n_threads` | `--threads` | 1 | Number of CPU threads used to run independent MC replicas in parallel. |
| `seed` | `--seed` | empty | Random seed of the Monte Carlo (empty = random). |
| `verbose` |  | 0 | Verbosity 0-3 (same as -v, -vv, -vvv in the command line). |

### Input (`[input]`)

| Key | Flag | Default | Meaning |
|---|---|---|---|
| `bedpe_path` | `--bedpe-file` | empty | A .bedpe file path with loops. It is required. |
| `rept_path` | `--rt-file` | empty | The replication timing dataset (.txt, .bw, .bigwig). If empty, no replication is modelled (unless USE_SC_REPT). |
| `sc_rept_path` | `--rt-file` | built-in file | The single cell replication timing dataset (.parquet). |
| `use_sc_rept` | `--rt-file builtin` | False | Use the built-in single-cell replication timing when REPT_PATH is empty. |
| `chrom` | `--region` | empty | Chromosome of the modelling region of interest. |
| `region_start` | `--region` | empty | Starting region coordinate. Empty = whole chromosome. |
| `region_end` | `--region` | empty | Ending region coordinate. Empty = whole chromosome. |
| `n_beads` | `--n-beads` | empty | Number of Simulation Beads. Empty = region length / 2 kb. |

### Pre-processing (`[preprocessing]`)

| Key | Flag | Default | Meaning |
|---|---|---|---|
| `rep_sim_t` |  | 1000 | Number of time points of the replication simulation. |
| `bigwig_bin_size` |  | 1000 | Bin size (bp) used to average a Repli-Seq bigWig. |
| `bigwig_sigma` |  | 200.0 | Gaussian smoothing (in bins) of a Repli-Seq bigWig. |
| `bigwig_log_norm` |  | True | Log-transform Repli-Seq signal to fix its skewness. |

### Replication (`[replication]`)

| Key | Flag | Default | Meaning |
|---|---|---|---|
| `rep_with_stress` | `--with-stress` | False | True sets REP_T_STD_FACTOR, REP_SPEED_SCALE and REP_INIT_RATE_SCALE to replication-stress values. |
| `rep_t_std_factor` |  | 0.1 | Factor multiplying the standard deviation of the replication timing curve. |
| `rep_speed_scale` |  | 10 | A scale that quantifies the speed of the replication forks. |
| `rep_init_rate_scale` |  | 1.0 | Multiplier of the experimentally estimated initiation rate. |
| `rep_start_time` |  | 15000 | MC step at which replication starts. |
| `rep_time_duration` |  | 20000 | Duration of replication (in MC steps). |

### Monte Carlo (`[mc]`)

| Key | Flag | Default | Meaning |
|---|---|---|---|
| `lef_rw` |  | True | Random walk for cohesins instead of unidirectional sliding. |
| `cohesin_blocks_condensin` |  | False | Allow cohesins to block condensins during simulation. |
| `random_init_spins` |  | True | Use a random initial distribution of spins. |
| `p_rew` |  | 0.5 | Probability of proposing a rewiring move (LEF move). |
| `rep_fork_push` |  | True | Hard-coded replication fork pushing in the MC moves. |
| `rep_fork_epigenetic_organizer` |  | True | Model replication forks as epigenetic organizers. |
| `n_steps` | `--n-steps` | 50000 | Total number of Monte Carlo steps. |
| `n_sweep` |  | 1000 | Number of node or link state changes proposed per simulation step. |
| `n_lef` |  | empty | Number of loop extrusion factors (LEFs). Empty = twice the number of CTCFs. |
| `n_lef2` |  | 0 | Number of second-family LEFs (different speed). |
| `mc_step` |  | 200 | Sampling frequency of MC steps (avoids autocorrelation). |
| `burnin` |  | 1000 | Burn-in period (steps before equilibrium). |
| `t_mc` |  | 1.5 | Temperature of the stochastic MCMC model. |
| `folding_coeff` |  | 1.0 | Folding coefficient for the first family of LEFs. |
| `folding_coeff2` |  | 0.0 | Folding coefficient for the second family of LEFs. |
| `rep_coeff` |  | 0.01 | Replication-barrier penalty (energy scale = coeff * 1e8; was 1.0 = 1e8 in the published v1.0). |
| `epi_interact_coeff` |  | 1.0 | Interaction coefficient of the epigenetic (Potts) state model. |
| `epi_field_coeff` |  | 1.0 | Field coefficient of the epigenetic (Potts) state model. |
| `cross_coeff` |  | 1.0 | Coefficient for LEF crossing. |
| `bind_coeff` |  | 1.0 | Coefficient for CTCF binding. |
| `n_replicas` | `--n-replicas` | 1 | Independent MC chains; each is processed into its own replica_XX sub-folder. |

### Molecular dynamics (`[md]`)

| Key | Flag | Default | Meaning |
|---|---|---|---|
| `initial_structure_type` |  | rw | rw, confined_rw, self_avoiding_rw, helix, circle, spiral, sphere, hilbert. |
| `simulation_type` | `--run-md / --md-mode` | empty | EM (many minimizations) or MD (one minimization + dynamics). Empty = no 3D structures. |
| `dcd_reporter` |  | False | Save a .dcd video of the 3D structure. |
| `integrator_type` |  | langevin | langevin or brownian. |
| `integrator_step` |  | 100 femtosecond | Integrator time step (e.g. '100 femtosecond'). |
| `forcefield_path` |  | built-in file | Path to XML file with the forcefield. |
| `ev_p` |  | 0.01 | Probability that excluded volume of a bead is randomly disabled. |
| `tolerance` |  | 1.0 | Stopping tolerance of the energy minimization. |
| `sim_temp` |  | 310 kelvin | Temperature of the 3D simulation (e.g. '310 kelvin'). |
| `sim_step` |  | 1000 | MD steps performed each time the loop forces change. |

### Output (`[output]`)

| Key | Flag | Default | Meaning |
|---|---|---|---|
| `save_plots` |  | True | Save diagnostic plots. |
| `save_mdt` |  | True | Save metadata of the stochastic simulation. |
| `report` | `--no-report` | True | Write report.html (model summary, figures, tables, command and all parameters). |
| `html_3d` | `--no-viewer` | True | Write an interactive 3D viewer (viewer_3d.html) of the ensemble. |
| `movie` | `--no-movie` | True | Write a PyVista movie of the ensemble (mp4 + gif, interpolated). |
| `movie_color` |  | auto | Movie colouring: auto, sister, compartment, rainbow or replication. |
| `movie_interp` |  | 5 | Interpolated frames between two structures in the movie. |
| `fig_dpi` |  | 200 | Resolution (dpi) of the saved figures and of the report. |
| `viz_heats` |  | True | Save the averaged heatmaps of the 3D ensemble. |

## Changing the physics in detail

| You want to change | Go to |
|---|---|
| Strength of loops, CTCF binding, crossing, replication barrier, Potts field and coupling, temperature | `[mc]`, see [mc_model.md](mc_model.md) |
| Fork speed, origin spread, replication stress, S-phase timing | `[replication]`, see [replication_model.md](replication_model.md) |
| Integrator, 3D temperature, excluded volume, starting shape, force field | `[md]`, see [md_model.md](md_model.md) |
| Figure resolution, movie and viewer settings | `[output]`, see [visualization.md](visualization.md) |

To change force-field terms beyond these parameters (bond strengths, compartment depths), edit the XML in `src/forcefields/` or the force definitions in `src/MD/forces.py`.

## In the code

| Function or class | What it does |
|---|---|
| `build_args()  (`config/arg_defs.py`)` | the table of every parameter: section, name, help, type and default |
| `Args, Arg, cast  (`config/arg_types.py`)` | `args.N_BEADS` style access and casting of strings to python types |
| `read_ini(args, path)` | reads a config file; an unknown key raises an error |
| `apply_stress(args)` | replication stress values used by `--with-stress` |
| `write_config(args, path)` | writes `config_used.ini` into the run folder |

To add a parameter, add one line `A(section, NAME, help, type, default)` in `src/config/arg_defs.py`; it then works in the ini file and in `args.NAME`.
