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
| run | `--n-replicas N` | Independent Monte Carlo replicas |
| run | `--seed N` | Random seed |
| output | `--out DIR` | Results root; each run gets its own sub-folder |
| output | `--name NAME` | Exact name of the run folder |
| output | `--platform CPU/CUDA/OpenCL` | OpenMM platform |
| output | `--threads N` | CPU threads for parallel replicas |
| output | `--no-report`, `--no-viewer`, `--no-movie` | Skip `report.html`, the 3D viewer or the movie |
| output | `-v`, `-vv`, `-vvv` | More log detail |

Options with a dash also accept an underscore: `--rt_file` is the same as `--rt-file`.

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
