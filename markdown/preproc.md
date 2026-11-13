# Pre-processing

Pre-processing turns a list of loops or CTCF peaks into the quantities the Monte Carlo needs: how strongly each bead attracts a loop extruder from the left and from the right.

## Input

A file of loops (`.bedpe`) or CTCF peaks (`.bed`, `.narrowPeak`) and a region; the columns and formats are listed in [Input data](input_data.md#ctcf-sites). A peak sits on one bead and adds $w(1-o)$ to $L$ and $w\,o$ to $R$ exactly as an anchor does (defined [below](#binding-potentials)). If no loops or peaks lie in the region, RepliSage stops with a clear message.

## Orientation from DNA (motif finder)

Sites without an orientation can get one from the genome sequence ([input format](input_data.md#reference-genome-and-motif)). RepliSage scans each anchor or peak, on both strands, for the JASPAR CTCF motif (MA0139.1, 19 bp) or your own PFM (`MOTIF_PFM`, a count matrix with rows A C G T).

* **Score.** A window of the sequence scores $s = \sum_j \log_2 \frac{P(x_j \mid \text{motif})}{1/4}$: how much more likely it is under the motif than under random DNA (higher = closer to the consensus).
* **Hit.** A window is a hit when $s$ passes the cut-off that random DNA reaches with probability `MOTIF_PVALUE` ($10^{-5}$ by default).
* **Orientation.** With several hits the site gets the reverse-strand share, each hit weighted by $2^{s}$:

$$o = \frac{\sum_{\text{reverse hits}} 2^{s}}{\sum_{\text{all hits}} 2^{s}}, \qquad o \in [0, 1]$$

A site without a hit stays unoriented, and orientations already in your file are never overwritten. `MOTIF_PAD` widens the scanned window on both sides.

## Smoothing

**CTCF profiles.** A site is a spike on one bead, but a real site is not exact to one bead and LEF legs move. As in LoopSage, $L$ and $R$ (defined [below](#binding-potentials)) are therefore blurred with a Gaussian of width $\sigma$ beads, so that sites become soft barriers:

$$\sigma = \max\left(1.5,\ N_{\text{beads}} / 300\right)$$

* Set it with `--smooth-sigma` (`SMOOTH_SIGMA`); turn it off with `--no-smooth` (`SMOOTH_INPUT = False`).
* Loops use a plain Gaussian. Peaks cap $\sigma$ at three times the median gap between sites (at least 3 beads), so isolated peaks are not erased.
* The total strength is kept (up to the region edges), so the binding energy keeps its scale.
* $N_{\text{CTCF}}$ and the "legs on a CTCF site" statistic use the raw sites (the latter with a margin of $\sigma$ beads). With fine beads $\sigma$ can be many kb; the log states it.

**Replication front.** `REP_FRONT_SMOOTH` (default 2 beads) smooths the *replication time* of each bead with a Gaussian and rebuilds the replicated/unreplicated state from it. The front becomes a gentle profile instead of a random staircase; the start and end of S phase are kept. `0` keeps it sharp.

## Hi-C map for validation

A `.hic` file ([input format](input_data.md#hi-c-map)) is read for the same region, put on the bead grid and uses it to [validate the model](analysis.md#hi-c-validation).

* **Bin size.** `HIC_RESOLUTION` if given, else the size closest to the bead size (finer preferred).
* **Normalisation.** `HIC_NORM` (KR by default); if the file lacks it, the next one is used (KR, SCALE, VC_SQRT, VC, NONE). The log reports any fallback.
* **Masked beads.** Beads with less than half of their bins in the file are left out of the comparison.

## From base pairs to beads

The region is cut into `N_BEADS` equal pieces. For a region of $G$ base pairs, one bead spans $G / N_{\text{beads}}$ base pairs. An anchor is assigned to the bead at the middle of its interval.

## Binding potentials

For every bead $k = 1, \dots, N_{\text{beads}}$ pre-processing computes two numbers, the **left potential** $L(k)$ and the **right potential** $R(k)$. They tell how strongly bead $k$ attracts the left leg $m_i$ and the right leg $n_i$ of a loop extruding factor (LEF) $i$. They enter the binding energy of the [Monte Carlo model](mc_model.md):

$$\mathcal{H}_{\text{bind}} = \mathcal{C}_{\text{bind}} \sum_{i=1}^{N_{\text{lef}}} \left(L(m_i) + R(n_i)\right)$$

Since $\mathcal{C}_{\text{bind}} < 0$, a LEF with legs on beads with large $L$ and $R$ lowers the energy, which pulls loops towards the CTCF loops of your data. Only the shape of $L$ and $R$ matters, not the scale of $w_\lambda$, because $\mathcal{C}_{\text{bind}}$ divides by $\sum_k (L(k) + R(k))$.

**Building $L$ and $R$.** Start from $L(k) = R(k) = 0$ and go through the loops $\lambda = 1, 2, \dots$ of the BEDPE file. Each loop brings three ingredients:

* two **anchor beads** $a_\lambda$ and $b_\lambda$, the beads that hold the two ends of the loop;
* a **strength** $w_\lambda \ge 0$, the score in column 7: how strongly the experiment supports this loop (for example the number of ChIA-PET read pairs or a HiChIP score). A strong loop makes its anchors more attractive for LEF legs, and a weak loop only slightly. If your file has no meaningful score, give every loop the same value, for example 1. A missing score counts as 1 and a negative one as 0;
* optionally, a **CTCF orientation** $o_c \in [0, 1]$ for each anchor $c$ (columns 8 and 9).

For each anchor bead $c \in \{a_\lambda, b_\lambda\}$ add:

* Without orientation information, the anchor attracts both legs equally:

$$L(c) \leftarrow L(c) + w_\lambda$$

$$R(c) \leftarrow R(c) + w_\lambda$$

* With the CTCF orientation $o_c \in [0, 1]$ from columns 8 and 9 (0 = forward only, 1 = reverse only):

$$L(c) \leftarrow L(c) + w_\lambda (1 - o_c)$$

$$R(c) \leftarrow R(c) + w_\lambda\, o_c$$

In words: a forward-facing CTCF site ($o_c = 0$) attracts the left leg of a LEF, and a reverse-facing site ($o_c = 1$) attracts the right leg; values in between split the attraction. An anchor whose orientation is unknown (negative, no motif) splits its strength evenly: $w_\lambda / 2$ to each of $L$ and $R$. Anchors from different loops that fall on the same bead add up, so beads that anchor many strong loops get the deepest potential wells.

The number of CTCF sites is the number of beads that carry an anchor in $L$ or $R$ after aggregation, $N_{\text{CTCF}} = \#\{k : L(k) > 0 \text{ or } R(k) > 0\}$, so anchors that fall into the same bead count once. If you do not set `N_LEF`, RepliSage uses $N_{\text{lef}} = N_{\text{CTCF}} / 2$ extruders, one per two CTCF sites. (Earlier versions used $2 N_{\text{CTCF}}$; with many CTCF sites per bead that can exceed the number of beads, and then no loops form, so set `N_LEF` below the number of beads, at most a quarter of it; at or above half of it is an error.)

The **input figures** show the anchors read from your file: a map (loop arcs coloured by convergent, tandem, divergent or uncertain configuration for BEDPE; anchor strength with orientation markers; orientation confidence $|2p-1|$, where $p$ is the probability of a reverse-oriented site), anchor statistics, and the binding profiles $L$ and $R$ before and after smoothing.

## Parameters

| Name | Where | Meaning | Default |
|---|---|---|---|
| `BEDPE_PATH` | `--bedpe-file`, `--input-file` | Loops (`.bedpe`) or peaks (`.bed`, `.narrowPeak`) | none (required) |
| `CHROM`, `REGION_START`, `REGION_END` | `--region` | Region to model | whole chromosome |
| `N_BEADS` | `--n-beads` | Number of beads | region length / 7 kb (max 10,000 beads) |
| `SMOOTH_INPUT`, `SMOOTH_SIGMA` | `--no-smooth`, `--smooth-sigma` | Gaussian smoothing of $L$ and $R$ | on, $\max(1.5, N/300)$ beads |
| `MOTIF_GENOME`, `MOTIF_PFM`, `MOTIF_PVALUE`, `MOTIF_PSEUDOCOUNT`, `MOTIF_PAD` | `--motif-genome` | Orientation from motifs | off, built-in CTCF motif, $10^{-5}$, 0.8, 0 bp |
| `REP_FRONT_SMOOTH` | | Smoothing of the replication front (`smooth_front` in `replication_simulation/front_smoothing.py`) | 2 beads |
| `HIC_FILE`, `HIC_NORM`, `HIC_RESOLUTION`, `HIC_TRACK_SMOOTH` | `--hic` | Hi-C map for validation; smoothing (beads) of its insulation and A/B tracks | none, KR, closest to the bead size, 7 |

Parameters for reading replication timing (`BIGWIG_BIN_SIZE`, `BIGWIG_SIGMA`, `BIGWIG_LOG_NORM`, `REP_SIM_T`) are in the [replication model](replication_model.md) and the [config page](config.md).

## Output

The potentials $L$ and $R$ are plotted in the report (section "Input") and used directly by the [Monte Carlo](mc_model.md).

## In the code

The stage is called from `run_preprocessing(ctx)` in - [`src/pipeline/preproc_stage.py`](https://github.com/SFGLab/RepliSage/blob/main/src/pipeline/preproc_stage.py): stage entry point, returns `(L, R, N_CTCF)`

| Function or class | What it does |
|---|---|
| `read_binding(path, region, chrom, n_beads, smooth, sigma, motif)` (`preproc/readers.py`) | reads BEDPE, BED or narrowPeak, fills missing orientations from the motif finder, smooths, and returns a `Binding` with `L`, `R`, the raw `L0`, `R0` and the number of sites |
| `preprocessing(bedpe_file, region, chrom, N_beads, smooth=False, sigma=None)` | the same as a tuple `(L, R, N_CTCF)` |
| `motif_orientation(chrom, starts, ends, genome, ...)` (`motif_finder/`) | probability of reverse orientation of each window from motif hits |
| `load_hic_region(path, geom, norm, resolution)` (`preproc/hic.py`) | Hi-C map on the bead grid |
| `binding_vectors_from_bedpe(bedpe_file, N_beads, region, chrom, normalization, viz)` | does the binning; `normalization` rescales the potentials, `viz` draws the input figure |
| `loop_stats(L, R)` | rows of the loop-input statistics table in the report |

Related arguments (see [Configuration](config.md)): `--bedpe-file` (`BEDPE_PATH`), `--region` (`CHROM`, `REGION_START`, `REGION_END`), `--n-beads` (`N_BEADS`).
