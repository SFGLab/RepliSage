<p align="center"><img src="assets/logo.png" alt="RepliSage" style="width:100%; max-width:100%;"></p>

<p align="center">
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-141414?logo=python&logoColor=white&labelColor=A50F2D"></a>
<a href="https://openmm.org/"><img alt="OpenMM" src="https://img.shields.io/badge/OpenMM-8.1%2B-141414?labelColor=A50F2D"></a>
<a href="https://numpy.org/"><img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.26-141414?logo=numpy&logoColor=white&labelColor=A50F2D"></a>
<a href="https://numba.pydata.org/"><img alt="Numba" src="https://img.shields.io/badge/Numba-0.60%2B-141414?labelColor=A50F2D"></a>
<a href="https://scipy.org/"><img alt="SciPy" src="https://img.shields.io/badge/SciPy-1.11%2B-141414?logo=scipy&logoColor=white&labelColor=A50F2D"></a>
<a href="https://matplotlib.org/"><img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-3.8%2B-141414?labelColor=A50F2D"></a>
<a href="https://pyvista.org/"><img alt="PyVista" src="https://img.shields.io/badge/PyVista-0.43%2B-141414?labelColor=A50F2D"></a>
<a href="https://docs.pytest.org/"><img alt="tested with pytest" src="https://img.shields.io/badge/tested%20with-pytest-141414?logo=pytest&logoColor=white&labelColor=A50F2D"></a>
<a href="https://github.com/SFGLab/RepliSage/blob/main/LICENSE"><img alt="License GPL v3" src="https://img.shields.io/badge/license-GPL%20v3-141414?labelColor=A50F2D"></a>
</p>

# RepliSage documentation

Welcome. These pages explain what RepliSage models, how to run it, and how to read what comes out. Start with the quick path below, then dive into whichever model you want to understand or tune.

## How the documentation is organised

| Section | What it covers |
|---|---|
| **Getting started** | [Installation](installation.md), [command line](cli.md) and [examples](examples.md) |
| **Theory** | The ideas behind the model, without implementation details: [scope and goals](scope.md), [the idea in a nutshell](intuition.md), [the Hamiltonian idea](hamiltonian.md), [Monte Carlo, MCMC and Metropolis](mcmc.md), [microstates and macrostates](micro_macro.md), [metrics](metrics.md) |
| **Implementation** | How the ideas are coded: [pipeline](pipeline.md), [pre-processing](preproc.md), [replication model](replication_model.md), [Monte Carlo model](mc_model.md), [molecular dynamics model](md_model.md) |
| **Reference** | [Input data](input_data.md), [configuration](config.md), [analysis](analysis.md) and [outputs and visualisation](visualization.md) |
| **Known limitations** | What the model leaves out: [limitations](limitations.md) |

## Quick path

1. Read [Scope and goals](scope.md) and [the idea in a nutshell](intuition.md) to see what the model is for.
2. [Install](installation.md) RepliSage and try the [command line](cli.md).
3. Look at the [pipeline](pipeline.md) to see what runs, and at the [outputs](visualization.md) to read the report.
4. Come back to the Theory pages when you want to understand or tune a parameter.

## Two ways to set things

* **Flags** (`--bedpe-file`, `--rt-file`, `--run-md`, `--with-stress`, ...) cover the everyday choices.
* **`config.ini`** (`-c config.ini`) holds everything else: energies, temperatures, force-field and plotting details. Flags always win over the file, and the file wins over the built-in defaults.

## A note on notation

The pages use the symbols of the RepliSage paper ($\mathcal{H}$, $N_{\text{beads}}$, $(m_i, n_i)$, $s_k$, $T_{\text{mc}}$, ...). The [notation table](mc_model.md#notation-paper-and-config) maps each one to its config name. Known simplifications are listed at the end in [Known limitations](limitations.md).

## A note on units

* Genomic positions are in base pairs. One bead covers a stretch of DNA of `(region length) / N_BEADS` base pairs.
* Time in the Monte Carlo and in the replication model is counted in MC steps.
* 3D coordinates are written in angstrom in the CIF files and simulated in nanometre units inside OpenMM.

## Citation

* Korsak, S. et al. "RepliSage: a stochastic graph-based framework for 3D chromatin modeling across the cell cycle." Nucleic Acids Research 54.12 (2026): gkag618.
* Buka, K. et al. "Improved cohesin HiChIP protocol and bioinformatic analysis for robust detection of chromatin loops and stripes." Communications Biology 8.1 (2025): 437.
* Massey, D. J. and Koren, A. "High-throughput analysis of single human cells reveals the complex nature of DNA replication timing control." Nature Communications 13 (2022): 2402.
