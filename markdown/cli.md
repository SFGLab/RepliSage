# Command line

```bash
replisage -h            # after `pip install .`
python src/run.py -h    # from the repository root
```

`-h` prints every option. The command line only carries the choices you make most often. Detailed physics and plotting parameters live in [`config.ini`](config.md).

## The options

| Group | Flag | Meaning |
|---|---|---|
| config | `-c, --config FILE` | Read parameters from this ini file |
| input | `--bedpe-file FILE` | Loops in BEDPE format (required unless the config file gives one) |
| input | `--rt-file FILE` | Replication timing: `.bigWig`/`.bw`, `.txt`, `.parquet` (single cell) or the word `builtin` |
| input | `--region CHR[:START-END]` | Region to model, for example `chr14:80835000-98674700`; a bare chromosome means the whole chromosome |
| input | `--n-beads N` | Beads per chain (default: region length divided by 2 kb) |
| run | `--run-md true/false` | Build 3D structures with OpenMM, or stop after the Monte Carlo |
| run | `--md-mode MD/EM` | Molecular dynamics or repeated energy minimisation |
| run | `--with-stress` | Replication stress (slower forks, more origins). Off by default |
| run | `--n-steps N` | Monte Carlo steps |
| run | `--mc-step N` | Record (thin) one MC state every N steps |
| run | `--burnin N` | MC steps discarded as burn-in |
| run | `--rep-start STEP`, `--rep-duration STEPS` | S phase start and length in MC steps (keep their sum within `--n-steps`) |
| run | `--md-steps N` | MD steps per recorded MC state |
| run | `--md-step-size TIME` | MD integrator step, for example `"100 femtosecond"` |
| run | `--n-replicas N` | Independent Monte Carlo replicas |
| run | `--seed N` | Random seed |
| output | `--out DIR` | Results root; each run gets its own sub-folder |
| output | `--name NAME` | Exact name of the run folder |
| output | `--platform CPU/CUDA/OpenCL` | OpenMM platform |
| output | `--threads N` | CPU threads for parallel replicas |
| output | `--no-report`, `--no-viewer`, `--no-movie` | Skip `report.html`, the 3D viewer or the movie |
| output | `-v`, `-vv`, `-vvv` | More log detail |

Options with a dash also accept an underscore: `--rt_file` is the same as `--rt-file`.

## Steps and sampling

Four numbers decide how long a run is and how much of it you keep. They can all be set from the command line.

| Flag | Symbol | Meaning | Default |
|---|---|---|---|
| `--n-steps` | $S$ | total Monte Carlo steps | 50000 |
| `--mc-step` | $\delta$ | one state is recorded every $\delta$ steps (thinning) | 200 |
| `--burnin` | $B$ | steps discarded at the start | 1000 |
| `--rep-start`, `--rep-duration` | $t_0$, $d$ | S phase runs from step $t_0$ to $t_0 + d$ | 15000, 20000 |
| `--md-steps` | $n_{md}$ | MD steps run for every recorded state | 1000 |
| `--md-step-size` | $\Delta t$ | MD integrator time step | 100 femtosecond |

The number of recorded states, and so of 3D structures, is

$$n = \left\lfloor \frac{S}{\delta} \right\rfloor$$

and the analysis uses the $n - B/\delta$ states after burn-in. For replication to finish inside the run keep $t_0 + d \le S$; the remaining steps are G2/M. The simulated MD time per structure is $n_{md}  \Delta t$.

Example: `--n-steps 30000 --mc-step 200 --burnin 1000` records 150 states and analyses 145 of them.

## What switches what on

| You provide | What runs |
|---|---|
| only `--bedpe-file` and a region | Loop extrusion and the report |
| plus `--rt-file` | Replication simulation, fork propagation, two sister chains in the model, Potts spins coupled to timing |
| plus `--run-md true` | 3D structures, the interactive viewer and the movie |

Without replication timing data RepliSage never runs the replication simulation or the fork propagation. With it, you decide whether the 3D part runs with `--run-md`.

## Precedence

Built-in defaults, then the config file, then the flags. For example `replisage -c my.ini --seed 3` uses everything from `my.ini` except the seed.

## Examples

```bash
# loop extrusion only
replisage --bedpe-file loops.bedpe --region chr14:80835000-98674700

# with replication timing, Monte Carlo only
replisage --bedpe-file loops.bedpe --region chr14:80835000-98674700 --rt-file GM12878.bigWig --run-md false

# with the shipped single-cell timing, 3D on a GPU, replication stress
replisage --bedpe-file loops.bedpe --region chr14:80835000-98674700 --rt-file builtin \
          --run-md true --platform CUDA --with-stress --n-replicas 3 --threads 3

# everything from a file
replisage -c my_config.ini
```

More in the [examples folder](examples.md).

## In the code

| Function or class | What it does |
|---|---|
| `main(argv)  (`src/run.py`)` | entry point of `replisage` and `python -m src.run` |
| `make_parser()` | builds the argument parser with the flags above |
| `flags_to_params(ns)` | turns the given flags into `PARAMETER: value` pairs |
| `get_config(argv, ini, overrides)` | resolves defaults, then the ini file, then flags; returns `(Args, config path)` |
| `parse_region(s)` | `'chr14:80,835,000-98,674,700'` becomes `(chrom, start, end)` |
| `str2bool(s)` | accepts `true`, `false`, `yes`, `no`, `1`, `0` |
| `build_context(args, make_dirs)` | decides which stages run from the inputs you gave |
| `run_pipeline(args, command)` | runs all stages |
