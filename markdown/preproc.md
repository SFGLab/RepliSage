# Pre-processing

Pre-processing turns a list of loops or CTCF peaks into the quantities the Monte Carlo needs: how strongly each bead attracts a loop extruder from the left and from the right.

## Input: a BEDPE file (loops)

A tab-separated file, one loop per line:

| Column | Content |
|---|---|
| 1 | Chromosome of the first anchor |
| 2, 3 | Start and end of the first anchor |
| 4 | Chromosome of the second anchor |
| 5, 6 | Start and end of the second anchor |
| 7 | Loop strength (for example ChIA-PET or HiChIP score); negative values are ignored |
| 8, 9 | Optional CTCF orientation of each anchor, between 0 and 1 (0 = forward only, 1 = reverse only, negative = no motif) |

Only loops that lie fully inside the chosen region and chromosome are kept. If none remain, RepliSage stops with a clear message.

## Other inputs: BED and narrowPeak

If you have CTCF ChIP-seq peaks instead of loops, give a `.bed` or `.narrowPeak` file. The format is chosen from the file extension.

| Format | Weight $w$ of a site | Position | CTCF orientation (optional) |
|---|---|---|---|
| `.bed` | column 5 (score; 1 if missing) | middle of the interval | column 7 = probability of reverse orientation, or the strand in column 6 (`+` = forward, `-` = reverse) |
| `.narrowPeak` | column 7 (signalValue; the score in column 5 if it is all zero) | the summit (column 10), else the middle | an extra column with the probability of *forward* orientation (column 11, or 6) |

A peak sits on one bead and adds $w (1 - o)$ to $L$ and $w\, o$ to $R$ exactly as an anchor does. Peak input has no loops, so the loop figures of the report and `PREFERENTIAL_BINDING` (which needs loops) are switched off.

## Orientation from the DNA sequence (motif finder)

If your sites have no orientation, give a reference genome with `--motif-genome genome.fa` (an uncompressed FASTA; an index `.fai` is created next to it). RepliSage scans every anchor or peak for the JASPAR CTCF motif (MA0139.1, or your own PFM in `MOTIF_PFM`) on both strands. A hit is kept when its score is at least the threshold that a random sequence reaches with probability `MOTIF_PVALUE` ($10^{-5}$). For a site with several hits the orientation is

$$o = \frac{\sum_{\text{reverse hits}} 2^{s}}{\sum_{\text{all hits}} 2^{s}},$$

where $s$ is the score of a hit: the share of the reverse strand, each hit weighted by its likelihood ratio $2^{s}$. A site without a hit stays without orientation. Orientations already in your file are kept. `MOTIF_PAD` widens the window scanned around each site.

## Smoothing

A site is a sharp spike on one bead, but a real CTCF site is not exact to one bead and the LEF legs move by thermal motion. Like LoopSage, RepliSage therefore smooths the profiles $L$ and $R$ with a Gaussian of width $\sigma$ beads (`SMOOTH_INPUT = True` by default):

$$\sigma = \max\left(1.5,\ N_{\text{beads}} / 300\right)$$

Change it with `SMOOTH_SIGMA` (`--smooth-sigma`), or switch smoothing off with `--no-smooth`. Loops are smoothed with a plain Gaussian. Peaks use a Gaussian whose width is capped (about three times the median gap between sites) so that isolated peaks are not erased. Smoothing keeps the total strength, so the binding energy does not change scale. The number of CTCF sites $N_{\text{CTCF}}$ and the "legs on a CTCF site" statistic always refer to the raw sites (the second one with a margin of $\sigma$ beads), not to the smoothed profile. For fine beads, $\sigma$ in kb can be large; the run log states it, and `SMOOTH_SIGMA` overrides it.

The **replication front** is smoothed too (`REP_FRONT_SMOOTH`, in beads, default 2): the replication time of each bead is averaged with its neighbours and the replicated/unreplicated state is rebuilt from it, so the front is a gently varying profile instead of a random staircase. The start and end of S phase are kept. Use 0 for the sharp front.

## Hi-C map for validation

With `--hic file.hic` the same region is read from the Hi-C file (needs `hic-straw`, `pip install hic-straw`), put on the bead grid and used to [validate the model](analysis.md#hi-c-validation). The bin size closest to the bead size and the normalisation `HIC_NORM` (KR by default) are tried first; if the file lacks them the next available ones are used, and the log says so.

## From base pairs to beads

The region is cut into `N_BEADS` equal pieces. For a region of $G$ base pairs, one bead spans $G / N_{\text{beads}}$ base pairs. An anchor is assigned to the bead at the middle of its interval.

## Binding potentials

For every bead $k = 1, \dots, N_{\text{beads}}$ pre-processing computes two numbers, the **left potential** $L(k)$ and the **right potential** $R(k)$. They tell how strongly bead $k$ attracts the left leg $m_i$ and the right leg $n_i$ of a loop extruding factor (LEF) $i$. They enter the binding energy of the [Monte Carlo model](mc_model.md):

$$\mathcal{H}_{\text{bind}} = \mathcal{C}_{\text{bind}} \sum_{i=1}^{N_{\text{lef}}} \left(L(m_i) + R(n_i)\right)$$

Since $\mathcal{C}_{\text{bind}} < 0$, a LEF with legs on beads with large $L$ and $R$ lowers the energy, which pulls loops towards the CTCF loops of your data. Only the shape of $L$ and $R$ matters, not the scale of $w_\lambda$, because $\mathcal{C}_{\text{bind}}$ divides by $\sum_k (L(k) + R(k))$.

**Building $L$ and $R$.** Start from $L(k) = R(k) = 0$ and go through the loops $\lambda = 1, 2, \dots$ of the BEDPE file. Each loop brings three ingredients:

* two **anchor beads** $a_\lambda$ and $b_\lambda$, the beads that hold the two ends of the loop;
* a **strength** $w_\lambda \ge 0$, the score in column 7: how strongly the experiment supports this loop (for example the number of ChIA-PET read pairs or a HiChIP score). A strong loop makes its anchors more attractive for LEF legs, and a weak loop only slightly. If your file has no meaningful score, give every loop the same value, for example 1. Loops with a missing or negative score are skipped;
* optionally, a **CTCF orientation** $o_c \in [0, 1]$ for each anchor $c$ (columns 8 and 9).

For each anchor bead $c \in \{a_\lambda, b_\lambda\}$ add:

* Without orientation information, the anchor attracts both legs equally:

$$L(c) \leftarrow L(c) + w_\lambda$$

$$R(c) \leftarrow R(c) + w_\lambda$$

* With the CTCF orientation $o_c \in [0, 1]$ from columns 8 and 9 (0 = forward only, 1 = reverse only):

$$L(c) \leftarrow L(c) + w_\lambda (1 - o_c)$$

$$R(c) \leftarrow R(c) + w_\lambda\, o_c$$

In words: a forward-facing CTCF site ($o_c = 0$) attracts the left leg of a LEF, and a reverse-facing site ($o_c = 1$) attracts the right leg; values in between split the attraction. An anchor with a negative orientation (no motif) adds nothing. Anchors from different loops that fall on the same bead add up, so beads that anchor many strong loops get the deepest potential wells.

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
| `REP_FRONT_SMOOTH` | | Smoothing of the replication front | 2 beads |
| `HIC_FILE`, `HIC_NORM`, `HIC_RESOLUTION` | `--hic` | Hi-C map for validation | none, KR, closest to the bead size |

Parameters for reading replication timing (`BIGWIG_BIN_SIZE`, `BIGWIG_SIGMA`, `BIGWIG_LOG_NORM`, `REP_SIM_T`) are in the [replication model](replication_model.md) and the [config page](config.md).

## Output

The potentials $L$ and $R$ are plotted in the report (section "Input") and used directly by the [Monte Carlo](mc_model.md).

## In the code

The stage is called from `run_preprocessing(ctx)` in - [`src/pipeline/preproc_stage.py`](https://github.com/SFGLab/RepliSage/blob/main/src/pipeline/preproc_stage.py): stage entry point, returns `(L, R, N_CTCF)`

| Function or class | What it does |
|---|---|
| `read_binding(path, region, chrom, n_beads, smooth, sigma, motif)` (`preproc/readers.py`) | reads BEDPE, BED or narrowPeak, fills missing orientations from the motif finder, smooths, and returns a `Binding` with `L`, `R`, the raw `L0`, `R0` and the number of sites |
| `preprocessing(bedpe_file, region, chrom, N_beads, smooth=False)` | the same as a tuple `(L, R, N_CTCF)` |
| `motif_orientation(chrom, starts, ends, genome, ...)` (`motif_finder/`) | probability of reverse orientation of each window from motif hits |
| `load_hic_region(path, geom, norm, resolution)` (`preproc/hic.py`) | Hi-C map on the bead grid |
| `binding_vectors_from_bedpe(bedpe_file, N_beads, region, chrom, normalization, viz)` | does the binning; `normalization` rescales the potentials, `viz` draws the input figure |
| `loop_stats(L, R)` | rows of the loop-input statistics table in the report |

Related arguments (see [Configuration](config.md)): `--bedpe-file` (`BEDPE_PATH`), `--region` (`CHROM`, `REGION_START`, `REGION_END`), `--n-beads` (`N_BEADS`).
