<p align="center"><img src="assets/logo.png" alt="RepliSage" style="width:100%; max-width:100%;"></p>

<p align="center">
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-141414?logo=python&logoColor=white&labelColor=A50F2D"></a>
<a href="https://openmm.org/"><img alt="OpenMM" src="https://img.shields.io/badge/OpenMM-8.1%2B-141414?labelColor=A50F2D"></a>
<a href="https://numpy.org/"><img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.26-141414?logo=numpy&logoColor=white&labelColor=A50F2D"></a>
<a href="https://numba.pydata.org/"><img alt="Numba" src="https://img.shields.io/badge/Numba-0.60%2B-141414?labelColor=A50F2D"></a>
<a href="https://scipy.org/"><img alt="SciPy" src="https://img.shields.io/badge/SciPy-1.11%2B-141414?logo=scipy&logoColor=white&labelColor=A50F2D"></a>
<a href="https://matplotlib.org/"><img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-3.8%2B-141414?labelColor=A50F2D"></a>
<a href="https://docs.pytest.org/"><img alt="tested with pytest" src="https://img.shields.io/badge/tested%20with-pytest-141414?logo=pytest&logoColor=white&labelColor=A50F2D"></a>
<a href="https://github.com/SFGLab/RepliSage/blob/main/LICENSE"><img alt="License GPL v3" src="https://img.shields.io/badge/license-GPL%20v3-141414?labelColor=A50F2D"></a>
</p>

# RepliSage documentation

Welcome. These pages explain what RepliSage models, how to run it, and how to read what comes out. Start with the quick path below, then dive into whichever model you want to understand or tune.

## Quick path

1. [Installation](installation.md): get RepliSage running.
2. [Command line](cli.md): the handful of flags you will use every day.
3. [Pipeline](pipeline.md): what runs, in which order, and which inputs switch which parts on.
4. [Outputs and visualisation](visualization.md): the report, the 3D viewer, the movie and the ChimeraX files.

## The models

| Page | What it covers |
|---|---|
| [Pre-processing](preproc.md) | From a BEDPE of loops to CTCF binding potentials on beads |
| [Replication model](replication_model.md) | Timing data to origins, forks and the replicated-DNA matrix |
| [Monte Carlo model](mc_model.md) | Loop extruders, epigenetic Potts spins and the energy function |
| [Molecular dynamics model](md_model.md) | Turning the sampled states into 3D structures with OpenMM |

## Configuring and analysing

| Page | What it covers |
|---|---|
| [Configuration](config.md) | `config.ini`: every advanced parameter, grouped by section |
| [Analysis](analysis.md) | Statistics, MCMC diagnostics and structural metrics |
| [Examples](examples.md) | Ready-to-run commands and annotated config files |

## Two ways to set things

* **Flags** (`--bedpe-file`, `--rt-file`, `--run-md`, `--with-stress`, ...) cover the everyday choices.
* **`config.ini`** (`-c config.ini`) holds everything else: energies, temperatures, force-field and plotting details. Flags always win over the file, and the file wins over the built-in defaults.

## A note on units

* Genomic positions are in base pairs. One bead covers a stretch of DNA of `(region length) / N_BEADS` base pairs.
* Time in the Monte Carlo and in the replication model is counted in MC steps.
* 3D coordinates are written in angstrom in the CIF files and simulated in nanometre units inside OpenMM.
