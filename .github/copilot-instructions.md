## RepliSage — Agent instructions for code changes

Purpose: give an AI coder the minimum, high-value knowledge to be productive in this repo.

- Big picture (3-stage pipeline)
  - Replication simulation: `Replikator.py` produces replication fork trajectories (Monte Carlo). Look here when changing initiation-rate logic or fork velocity.
  - Stochastic co-evolution: `RepliSage/stochastic_model.py` (and `energy_functions.py`, `initial_structures.py`) model LEFs, Potts compartments, and their energy terms. Edits to the MCMC or Potts energies belong here.
  - Molecular dynamics: OpenMM-based MD step implemented via `RepliSage/run.py` → `sim.run_openmm()` (uses `forcefields/` XMLs). Changes to reporters, integrators, or forcefield wiring belong here.

- Entry points & workflows
  - CLI entrypoint: the package exposes a console script `replisage` (setup.py entry_points). You can also run: `python -m RepliSage.run -c config.ini`.
  - Bulk runs: `run_all_chromosomes.py` iterates chromosomes by modifying `config.ini` and calling the module entrypoint. Use this for chromosome-scale bench/CI stubs.
  - Docker: the `Dockerfile` builds a venv and installs `pyRepliSage`. Useful for reproducing CUDA/OpenMM environments. The README contains a tested `docker run` example.

- Important files and what they show (quick map)
  - `README.md` — overall design, data formats and example Docker+CLI commands. Always consult it for required input formats (BEDPE, replication timing `.mat`/`.txt`).
  - `setup.py` and `requirements.txt` — required packages and minimum Python (>=3.10). OpenMM + openmm-cuda are required for GPU MD.
  - `RepliSage/args_definition.py` — canonical parameter list and defaults. Preferred pattern: set defaults here, then override via config file or CLI.
  - `RepliSage/run.py` — orchestrates the whole pipeline: config parsing, stochastic simulation launch, optional MD call `run_openmm()`, plotting and metadata save.
  - `RepliSage/energy_functions.py` — computational kernels for energy terms; be cautious modifying these (performance sensitive and sometimes cached/compiled).

- Project conventions and patterns
  - Configuration precedence: defaults (in code) ← config.ini values ← CLI args. `run.py` uses `args_definition` to convert to Python types and write final args back to disk.
  - Config keys are uppercase and placed under a `[Main]` section in `config.ini`. Example keys: `N_BEADS`, `N_LEF`, `REPT_PATH`, `SC_REPT_PATH`, `SIMULATION_TYPE`, `PLATFORM`.
  - Outputs: everything under `OUT_PATH` and subfolders `metadata`, `ensemble`, `plots`. Tests or dev runs should use `tmp/` to avoid polluting results.
  - Data expectations: loop files are `.bedpe` with probabilistic right/left columns; replication timing can be single-cell `.mat` or `.txt`. `run.py` validates `REPT_PATH` must be `.txt` when provided.
  - Logging: the code prints colored terminal messages (ANSI sequences). Small CLI runs are normal for dev debugging.

- Integration & external dependencies to be aware of
  - OpenMM (+ openmm-cuda) and CUDA are required for MD. Use Dockerfile or local CUDA toolkits to reproduce runtime.
  - `pyarrow` / `fastparquet` are used for Parquet I/O (single-cell inputs). `mat73` may be required for `.mat` handling.
  - External utility: motif finding is referenced from `SFGLab/3d-analysis-toolkit` (`find_motifs.py`) — used to populate BEDPE motif probabilities.

- Quick dev/run examples (from README)
  - Install deps: `pip install -r requirements.txt` or use the provided Docker image.
  - Run CLI: `replisage -c config.ini` or
    `python -m RepliSage.run -c config.ini`.
  - For quick multi-chromosome dev runs, `run_all_chromosomes.py` updates `config.ini` and invokes the module per chromosome.

- When editing code, pay attention to these hotspots
  - Any change to Monte Carlo moves, energy coefficients or Potts interactions → update `energy_functions.py`, `stochastic_model.py`, and test locally with a tiny `N_BEADS` and very few steps (`N_STEPS` small) to observe behavior.
  - Changes to CLI/config handling → `RepliSage/run.py` and `args_definition.py` (the latter is the source of truth for names/types).
  - MD changes → `initial_structures.py`, `md_model.py`, `forcefields/*` and `run_openmm()` in `run.py`.

- Helpful examples to copy/paste when modifying code
  - Quick small sanity run (dev): set `N_BEADS=200`, `N_LEF=20`, `N_STEPS=1000`, `SIMULATION_TYPE=` (or omit MD), `OUT_PATH=tmp/test_run`, then run `python -m RepliSage.run -c config.ini`.

If anything in these instructions is unclear or you want the file to include CI/test guidance or repo-specific unit-test examples, tell me which area to expand and I will update the document.
