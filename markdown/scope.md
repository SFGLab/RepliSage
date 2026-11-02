# Scope and goals

RepliSage follows one chromosome through the **whole cell cycle** (G1, S, G2/M) in a single model, with **S phase as its focus**. Most tools treat loop extrusion, compartments or replication separately; here the three can influence each other, which lets you test mechanistic ideas with your own data. The paper calls it a proof-of-concept framework, and that is a good way to use it.

## Goals

* Simulate all phases and obtain a 3D **ensemble** for each, from your own loops (BEDPE) and replication timing.
* Study **S phase**: how moving forks affect loops, epigenetic states and sister chromatids.
* Test hypotheses: fork speed, origin firing, barrier strength, whether forks organise epigenetic states.
* Model **replication stress** (RS): slower forks and more origins firing.

## The phases

| Phase | What the model does |
|---|---|
| G1 | One DNA copy. LEFs (cohesin) extrude loops, epigenetic states follow the timing-derived field |
| S | Replikator provides fork positions. Forks push LEFs and may organise states. Two chains are tied where DNA is copied and gradually separate |
| G2/M | Replication ties are off. A second LEF family (condensin, more processive) and random excluded-volume relaxation (topoisomerase) are added; a gentle force separates the sisters |

## What the paper reports

* Loops get shorter in S phase and shorter still under RS, as in single-cell Hi-C. Cohesin HiChIP under mild RS (aphidicolin, hydroxyurea) shows the same.
* Compartments stay largely stable under RS. The model's epigenetic states correlate with Hi-C compartments (above 80%).
* If forks organise epigenetic states, RS gives more B to A switching.
* The chromosome decompacts in S and compacts in G2/M, as in NucDynamics models from single-cell Hi-C.

Paper settings for chromosome-wide runs: 10,000 beads, 1000 cohesin and 1000 condensin LEFs, $T_{\text{mc}} = 1.8$, $2\times10^5$ Monte Carlo steps. The defaults in `config.ini` are lighter.

For the formalism behind the model, see [the Hamiltonian idea](hamiltonian.md). Please read [Known limitations](limitations.md) before drawing conclusions from a result.
