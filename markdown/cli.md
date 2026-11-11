# Command line

```bash
replisage -h            # after `pip install -e .` (run it in the repository root)
python -m src.run -h    # without installing, from the repository root
```

`-h` prints every option. The command line only carries the choices you make most often. Detailed physics and plotting parameters live in [`config.ini`](config.md).

## The options

| Group | Flag | Meaning |
|---|---|---|
| config | `-c, --config FILE` | Read parameters from this ini file |
| input | `--bedpe-file FILE` (or `--input-file`) | CTCF input: loops (`.bedpe`) or peaks (`.bed`, `.narrowPeak`); required unless the config file gives one |
| input | `--smooth-sigma BEADS`, `--no-smooth` | Width of the Gaussian smoothing of the CTCF input (default `max(1.5, n_beads / 300)`, as in LoopSage) or switch it off |
| input | `--motif-genome FASTA` | Reference genome: read CTCF orientation from the motif where the input has none |
| input | `--hic FILE` | Hi-C map (`.hic`) to validate the model against (needs `hic-straw`) |
| input | `--rt-file FILE` | Replication timing: `.bigWig`/`.bw`, `.txt`, `.parquet` (single cell) or the word `builtin` |
| input | `--region CHR[:START-END]` | Region to model, for example `chr14:80835000-98674700`; a bare chromosome means the whole chromosome |
| input | `--n-beads N` | Beads per chain (default: region length / 7 kb, about 10 beads per CTCF site) |
| run | `--run-md true/false` | Build 3D structures with OpenMM (default true), or stop after the Monte Carlo (false) |
| run | `--md-mode MD/EM` | Molecular dynamics (default: gives the replication trajectory) or repeated energy minimisation |
| run | `--with-stress` | Replication stress (slower forks, more origins). Off by default |
| run | `--preferential-binding` | LEFs load preferentially inside loop-spanned (CTCF-occupied) areas and rarely in loop-naked areas. Off by default |
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
| output | `--platform auto/CUDA/OpenCL/CPU` | OpenMM platform (default auto: CUDA, then OpenCL, then CPU; any failure falls back to the next) |
| output | `--threads N` | CPU threads for parallel replicas |
| output | `--no-report`, `--no-viewer` | Skip `report.html` or the 3D viewer |
| output | `--skip-movies [true/false]` | Skip the movies: they are written by default but take minutes |
| output | `-v`, `-vv`, `-vvv` | Log detail. Without a flag: stages, progress bars, successes (results saved) and warnings only. `-v` adds the input and output tables, `-vv` the detailed information messages, diagnostics and library warnings, `-vvv` everything (including low-level messages of the graphics libraries, which are hidden otherwise) |

Options with a dash also accept an underscore: `--rt_file` is the same as `--rt-file`. A parameter that has no flag (for example `--n_lef`) gives a clear error that points you to the config file. A mistyped flag gets a "did you mean" hint, and a missing BEDPE file or a malformed region stops the run with a message naming the problem.

## Steps and sampling

Four numbers decide how long a run is and how much of it you keep. They can all be set from the command line.

| Flag | Symbol | Meaning | Default |
|---|---|---|---|
| `--n-steps` | $S$ | total Monte Carlo steps | 150000 |
| `--mc-step` | $\delta$ | one state is recorded every $\delta$ steps (thinning) | 200 |
| `--burnin` | $B$ | steps discarded at the start | 5000 |
| `--rep-start`, `--rep-duration` | $t_0$, $d$ | S phase runs from step $t_0$ to $t_0 + d$ | 50000, 50000 |
| `--md-steps` | $n_{md}$ | MD steps run for every recorded state | 5000 |
| `--md-step-size` | $\Delta t$ | MD integrator time step | 100 femtosecond |

The number of recorded states, and so of 3D structures, is

$$n = \left\lfloor \frac{S}{\delta} \right\rfloor$$

and the analysis uses the $n - B/\delta$ states after burn-in. For replication to finish inside the run keep $t_0 + d \le S$; the remaining steps are G2/M. The simulated MD time per structure is $n_{md}  \Delta t$.

Example: `--n-steps 150000 --mc-step 200 --burnin 5000` records 750 states and analyses 725 of them.

## What switches what on

| You provide | What runs |
|---|---|
| only `--bedpe-file` and a region | Loop extrusion, 3D structures (MD) and the report |
| plus `--rt-file` | Replication simulation, fork propagation, two sister chains in the model, epigenetic states coupled to timing |
| `--run-md false` | Stop after the Monte Carlo: no 3D structures, viewer or movie |

Without replication timing data RepliSage never runs the replication simulation or the fork propagation. The 3D part (molecular dynamics, with the interactive viewer and the movie) runs by default, because the aim is the replication trajectory; use `--run-md false` for a quick Monte Carlo only run, or `--md-mode EM` for a faster, static version. The 3D platform is chosen automatically: CUDA, then OpenCL, then CPU, falling back on any error (`--platform` forces one).

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

# everything from a file, no other arguments
replisage -c my_config.ini
```

Every option has a key in the config file, and the file has many more (MC, replication, MD, force field, figures). The command line is only a shortcut for the everyday ones. A file with `bedpe_path`, `chrom` (+ region) and `rept_path` is enough to run; `rept_path` takes a file, `builtin` or `none`. The full template is `src/config.ini`.

## Safety checks

Bead size defaults to 7 kb (about 10 beads per CTCF site at typical anchor density; it scales with region length, not with the CTCF count, so dense CTCF clusters collapse at low resolution on purpose). Regions that would need more than 10,000 beads (the paper's chromosome-wide size) are coarsened. Bead sizes outside 200 bp - 50 kb are an error, and outside 1 - 10 kb a warning. `N_LEF + N_LEF2` at or above half the number of beads is an error (default: half the CTCF sites, at most N/4).

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
