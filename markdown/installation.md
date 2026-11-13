# Installation

RepliSage needs Python 3.10 or newer and is tested on Linux. The 3D stage uses OpenMM, which is much faster on a GPU, while the Monte Carlo runs on the CPU.

## From PyPI

```bash
pip install pyRepliSage
replisage -h
```

This installs the `replisage` command. The package is imported as `RepliSage`.

If `replisage -h` shows old options (for example `--region_start`), an older installed copy is shadowing the repository. Reinstall with `pip install -e .` in the repository root, or run `python -m src.run` from there.

## From source

```bash
git clone https://github.com/SFGLab/RepliSage
cd RepliSage
pip install -r requirements.txt     # dependencies
pip install -e .                    # the replisage command, editable
```

Without installing, you can also run it from the repository root: `python -m src.run -h`.

## A clean environment (recommended)

```bash
conda create -n replisage python=3.11
conda activate replisage
pip install pyRepliSage
```

## Optional parts

| You want | Install |
|---|---|
| The pyvista movies (`movie_3d_sister.mp4`, `movie_3d_compartment.mp4`) | `pip install "pyRepliSage[viz]"` (the `ffmpeg` encoder comes with `imageio-ffmpeg`) |
| Hi-C validation (`--hic`) | `pip install "pyRepliSage[hic]"` (installs `hic-straw`) |
| 3D on a GPU | a CUDA or OpenCL OpenMM build, then `--platform CUDA` (or `OpenCL`) |
| To run the tests | `pip install "pyRepliSage[test]"`, then `pytest` (fast tests) or `python run_tests.py` (all checks with one summary) |
| To build these docs | `pip install "pyRepliSage[docs]"`, then `mkdocs serve` |

The interactive `viewer_3d.html` needs nothing extra: it is one self-contained file that opens in any browser.

## Check that it works

```bash
replisage -h                                   # lists every flag
replisage --bedpe-file loops.bedpe --region chr14:80835000-98674700 --n-steps 5000
```

The second command is a short loop-extrusion run. It writes a folder with `report.html` that you can open in a browser.

## Docker

A `Dockerfile` is in the repository: `docker build -t replisage .`

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| The first Monte Carlo run pauses for a few seconds | numba compiles the energy functions once; later runs start at once |
| `No template found for residue` in the 3D stage | the force field file does not match the structure; keep the default `FORCEFIELD_PATH` |
| `pyBigWig` fails to install | install it from conda-forge: `conda install -c conda-forge pybigwig` |
| The movie is skipped | install `pyvista`; use `--skip-movies` to skip the movies |
| GPU not used | check `--platform`, and that your OpenMM sees the device: `python -m openmm.testInstallation` |

Next: the [command line](cli.md) and [examples](examples.md).
