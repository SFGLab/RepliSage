"""Texts of the report: model summary and a short caption (what it shows / what to expect) per figure."""

TAGLINE = "Loop extrusion × DNA replication × chromatin compartments"

ABOUT = [
    "RepliSage simulates how chromatin loops, DNA replication and compartments evolve together through the cell cycle. "
    "Loop extruding factors (LEFs, e.g. cohesin) grow loops until they meet CTCF sites. Replication forks move through the DNA "
    "and act as moving barriers for LEFs. Every bead also carries an epigenetic state that is spread behind the replication fork "
    "and organises A/B compartments.",
    "A Monte Carlo (Metropolis) simulation produces the time evolution of loops, replication and epigenetic states. "
    "Molecular dynamics (OpenMM) can then turn each Monte Carlo state into a 3D structure, with two sister chains once replication is modelled.",
]
STAGES = [("Loop extrusion", "LEFs with two legs (m, n) bind CTCF sites and extrude loops."),
          ("Replication", "Origins fire from the initiation rate I(x,t); forks move in both directions."),
          ("Compartments", "A 5-state Potts spin per bead (−2…2): A-like if positive, B-like if negative."),
          ("3D structure", "OpenMM builds the chromatin chain(s) from every sampled state.")]
EQ = [("E = E<sub>le</sub> + E<sub>rep</sub> + E<sub>potts</sub>", "Total energy; a move is accepted with probability min(1, e<sup>−ΔE/T</sup>) (T = MC temperature)."),
      ("E<sub>le</sub> = c<sub>fold</sub> Σ log(n<sub>i</sub>−m<sub>i</sub>) + c<sub>cross</sub> Σ K + c<sub>bind</sub> Σ [L(m<sub>i</sub>)+R(n<sub>i</sub>)]",
       "Folding favours long loops, K penalises crossing LEFs, L and R reward legs on left / right CTCF sites."),
      ("E<sub>rep</sub> = c<sub>rep</sub> Σ ℛ(m<sub>i</sub>, n<sub>i</sub>; f<sub>rep</sub>)", "Penalty for loops that straddle a replication fork (f<sub>rep</sub> = replicated or not)."),
      ("E<sub>potts</sub> = c<sub>p1</sub> Σ h<sub>k</sub> s<sub>k</sub> + c<sub>p2</sub> Σ J<sub>ij</sub> |s<sub>i</sub>−s<sub>j</sub>|",
       "h = field from replication timing; J<sub>ij</sub> counts backbone and loop links between beads i and j.")]
READ = "Grey band = burn-in (discarded). Red band = S phase (replication). Red = A compartment (positive spin), blue = B compartment (negative spin). Orange / teal = left / right LEF legs and CTCF sites."

# fid -> (title, what it shows, what to expect)
CAP = {
 "loops_input": ("CTCF input", "Left (orange, up) and right (teal, down) CTCF binding strength of every bead, from your loop file.", "Peaks mark loop anchors where LEFs stall. Few peaks mean little loop structure."),
 "mc_energy": ("Energies", "Total, folding and binding energy over MC steps: trace (grey) and moving average (black).", "Stable fluctuations after burn-in. With replication, a shift can appear in S phase as forks restrict loops."),
 "mc_energy_potts": ("Potts energy", "Epigenetic (Potts) energy and magnetization over MC steps.", "Relaxation in G1, then a change in S phase as replication timing switches states; settling in G2/M."),
 "mc_distributions": ("Distributions", "Violin per phase (median dot, box = middle 50 %) of energy, folding and loop length. One histogram without replication.", "Overlapping violins mean the phase barely matters; shifted ones show the effect of replication."),
 "mc_convergence": ("Chain diagnostics", "Geweke z (start vs end of chain), effective sample size share (ESS/n) and split R̂ for each observable.", "|z| < 2, R̂ < 1.01 and ESS/n of several percent mean a well-mixed chain. Red bars flag problems: run longer."),
 "mc_replicas": ("Replica comparison", "Same observables across independent replicas (different seeds), with split R̂ between them.", "Replicas overlap; R̂ close to 1 means they agree."),
 "movie_gif": ("3D movie", "Interpolated 3D structures with the contact map (moving-window average) and the cell-cycle bar. Orange / gold = sister chromatids; black = LEF anchors; teal = forks.", "Compact G1 chromatin; two sisters appear during S and drift apart in G2/M; loops (black) hold beads together."),
 "mc_acf": ("Autocorrelation", "How quickly samples forget their past. τ = integrated autocorrelation time (samples).", "Curves fall to zero within a few lags. Large τ means correlated samples and fewer independent ones."),
 "mc_running": ("Running mean", "Mean after burn-in as samples accumulate. Band = final mean ± 2 Monte Carlo standard errors.", "The curve settles inside the band. A drift means the run is too short."),
 "rep_violations": ("Fork conflicts", "Number and strength of loops straddling a replication fork during S phase.", "Low when the replication penalty is strong. Bursts appear as forks push LEFs."),
 "lef_traj": ("LEF trajectories", "Each red/black/grey shade is one LEF; the band spans its two legs over time. Dashed line = replication front; bars on the right = CTCF sites (red left, black right).", "Long-lived loops at CTCF anchors, nested bands, and loops rearranged behind the front in S phase."),
 "lef_coverage": ("Loop coverage", "Number of loops over every bead (white = none, black = many). Right: time average.", "Dark bands at strong domains and gaps between them; a stable profile after burn-in."),
 "loop_length": ("Loop length", "Mean loop length with 95 % CI across LEFs (left); distribution per phase (right).", "Constant after burn-in. A shift in S phase means forks shorten or reshape loops."),
 "violin_loop_length": ("Loop length by phase", "Distribution of all loop lengths in G1, S and G2/M.", "Similar shapes if replication does not affect loops."),
 "lef_binding": ("LEF anchoring", "Top: share of bound LEFs and of legs on CTCF sites. Bottom: legs per bead for left (orange) and right (teal) legs, triangles = CTCF.", "Leg occupancy peaks at matching CTCF sites; a stronger binding coefficient raises the on-CTCF share."),
 "rep_forks": ("Replication forks", "Left: replicated DNA (green) in genome × time. Right: active forks (blue) and inferred origins (red triangles).", "Forks spread from origins in both directions; early regions replicate first; everything is replicated at the end."),
 "rep_fraction": ("Replicated fraction", "Fraction of the region replicated over S phase (T10, T50, T90 marked) and number of active forks.", "A smooth S-shaped curve; forks rise as origins fire and fall as forks merge."),
 "rep_initiation": ("Initiation rate", "I(x,t): chance that an origin fires at position x and time t. Top: where origins fire. Right: when.", "Diagonal bands: early regions fire early. Triangles (origins) sit on dark bands."),
 "rep_timing": ("Timing check", "Replication earliness from the data and from the simulation along the region (left); bead by bead (right) with Spearman ρ.", "Curves overlap and ρ is close to 1."),
 "epi_traj": ("Epigenetic states", "Potts spin of every bead over time: red = A-like (+), blue = B-like (−). Dashed line = replication front. Long runs are averaged in time bins.", "Blocks of equal colour in G1. In S phase colours change behind the front."),
 "epi_sorted": ("States by timing", "Same map with beads ordered from early to late replicating.", "If compartments follow timing, early beads stay A-like and late ones B-like; changes start at the early end."),
 "epi_fractions": ("State fractions", "Top: share of beads in each spin state (blue = B-like, red = A-like). Bottom: A (spin > 0) and B (spin < 0).", "Steady in G1, a shift during S phase, settling in G2/M."),
 "epi_switching": ("Compartment switching", "Fraction of beads switching B→A (red) or A→B (blue) over time; right: beads moving between B, N and A from the start to the end of S.", "Switching peaks in S phase. More B→A than A→B means A grows during replication."),
 "epi_switch_bars": ("S-phase switching", "Left: % of beads in B, N, A at the start and end of S. Right: % that switched (B→A red, A→B blue) or stayed.", "Most beads stay. The gap between B→A and A→B gives the net direction."),
 "epi_switch_map": ("Switching map", "Along the region: replication earliness, compartments at S start and S end, and switched beads (red B→A, blue A→B).", "Early regions tend to be A, late ones B. Switches mark where this changed during S."),
 "epi_same_sign": ("Same-sign loops", "Share of loops whose two legs have the same (teal) or opposite (orange) epigenetic sign; right: per phase.", "High share means loops stay inside one compartment."),
 "violin_same_sign": ("Same sign by phase", "Distribution of the same-sign loop share in G1, S and G2/M.", "A shift in S phase means loops link different states."),
 "epi_order": ("Order parameters", "Magnetization (largest state share), Binder cumulant and graph synchronization over time.", "Higher values as compartments form."),
 "epi_order2": ("Order parameters (2)", "Cluster order (largest linear cluster) and entropy of the state distribution.", "Entropy falls and cluster order rises as domains order."),
 "md_dynamics": ("MD energies", "Potential and kinetic energy (left) and temperature (right) of the last MD run.", "Energies settle and the temperature stays near the target."),
 "md_structure": ("3D size and shape", "Radius of gyration, mean pair distance and end-to-end distance of each sampled 3D structure.", "Stable fluctuations; compaction visible as drops."),
 "md_structure2": ("3D shape and contacts", "Asphericity, number of contacts and convex-hull volume of each sampled structure.", "Stable fluctuations after equilibration."),
}

# new diagnostic figures: (title, shows, expect) and a longer "how to read it" text
CAP.update({
 "mc_stationarity": ("Stationarity check", "Density of each observable in the first and in the second half of the stationary part of the run.", "Two overlapping curves: the chain has settled, so its statistics do not depend on when you look."),
 "mc_mcse": ("Error vs samples", "Left: Monte Carlo standard error relative to the spread (MCSE / SD) as samples accumulate. Right: effective samples, the number of independent draws the chain is worth.", "Both curves follow the dashed line at first and bend away from it: samples are correlated, so each one is worth less than an independent draw."),
 "mc_rolling": ("Rolling statistics", "Rolling mean with a band of one standard deviation for energy and loop length, and the rolling autocorrelation time of the energy.", "Flat mean and band in G1; a change at S phase; a stable autocorrelation time."),
 "mc_correlation": ("Observable correlations", "Left: rank correlation between all observables after burn-in. Right: energy against mean loop length, coloured by time.", "Folding energy follows loop length strongly; the colour gradient shows how the system moves between phases."),
 "mc_budget": ("Energy budget", "Left: the energy terms over time (Other = loop crossing and replication barrier). Right: mean of each term in G1, S and G2/M.", "Folding and binding dominate and stay steady; Other stays near zero if loops avoid crossing and respect forks."),
 "lef_dynamics": ("LEF binding dynamics", "Left: fraction of LEFs bound over time. Right: how long LEFs stay bound before they unbind, on a log axis.", "A steady bound fraction and a broad lifetime distribution with a long tail (stable CTCF-anchored loops)."),
 "epi_domains": ("Compartment domains", "Left: sizes of stretches of beads with the same compartment sign, per phase. Right: mean domain size over time.", "Domains of a few to tens of beads; S phase can fragment or merge them as timing and loops compete."),
})
HOW = {
 "mc_energy": "Faint line: every recorded sample. Bold line and band: moving average and one standard deviation. Grey = burn-in, red band = S phase. After burn-in the bold line should wobble around a flat level. A steady drift means the run is too short; raise BURNIN or N_STEPS. A jump at the S-phase edges is expected: forks change the allowed loops.",
 "mc_energy_potts": "Potts energy measures how well spins agree with the replication timing field and with their loop partners; magnetisation is the mean spin (A positive, B negative). In G1 both should be steady. Their change in S phase shows epigenetic colour reorganising as DNA replicates.",
 "mc_distributions": "Each violin is a smoothed histogram of one phase; the white dot is the median and the thick bar the interquartile range. Phases whose violins barely overlap differ clearly. Without replication a single histogram is shown. Wide violins mean large fluctuations, not an error.",
 "mc_convergence": "Left: Geweke z compares the start and the end of the stationary part; bars inside the grey band (between -2 and 2) pass. Middle: effective sample size as a percentage of the samples; low values mean strongly correlated samples, so increase MC_STEP or run longer. Right: split R-hat; below 1.01 (teal) is good, gold is borderline, red means the halves of the chain disagree.",
 "mc_acf": "Autocorrelation is the correlation of a series with itself shifted by a lag. It starts at 1 and should die out. The number tau in the legend is the integrated autocorrelation time: roughly how many consecutive samples carry the information of one independent sample. If the curve stays high for tens of lags, raise MC_STEP so that recorded states are further apart.",
 "mc_running": "The line is the average of all samples up to that step. It should settle onto the horizontal line, inside the grey band of plus or minus two Monte Carlo standard errors. If it keeps sliding the estimate is not converged. The faint line behind is the raw series.",
 "mc_stationarity": "A small KS statistic (Kolmogorov-Smirnov distance, 0 = identical, 1 = completely different) means the two halves look alike. Values above about 0.2 with visibly shifted curves say the chain was still moving: lengthen the burn-in.",
 "mc_mcse": "For independent samples the error falls as 1 over the square root of n (dashed line). Correlated chains fall more slowly, and the gap between the curves and the dashed line is the price of correlation. On the right, the effective sample size should keep growing as you add samples; if it flattens, more steps do not help and something is stuck.",
 "mc_rolling": "Moving windows reveal slow changes that a global average hides. The third panel estimates the autocorrelation time in each window: a sudden rise means the dynamics became sluggish (for example, loops locked in place by forks).",
 "mc_correlation": "Rank (Spearman) correlation is 1 when two observables rise together, -1 when one rises as the other falls, and 0 when unrelated. Strong blocks tell you which observables carry the same information, so you can pick one to monitor. Colour in the scatter runs from early to late steps: separate colour clouds show distinct phases.",
 "mc_budget": "The total energy is the sum of folding, binding, Potts and the rest. Terms with large swings drive the dynamics; terms near zero are not constraining the system. The bars compare phases: a bar that changes in S phase is the part of the model that responds to replication.",
 "lef_dynamics": "Bound means a LEF has legs on two different beads. A lifetime is the time between a LEF binding and unbinding. Short lifetimes mean frequent rebinding (loops form and disappear); a long tail means some loops are stable, typically anchored at CTCF sites. In S phase, forks may shorten lifetimes by pushing loops away.",
 "epi_domains": "A domain is an uninterrupted run of beads with the same compartment sign (A or B); neutral beads end a run. Compare the three curves: if S phase has more small domains, replication fragments the compartments; if G2/M has larger ones, the new copies have merged into bigger domains.",
}
def how_for(fid): return HOW.get(fid, "")

def cap_for(fid):
    if fid in CAP: return CAP[fid]
    if fid.startswith("md_heat_"): return ("Contact map", "Mean of 1/(distance + 1) between beads over the structures of one phase (dark = close).", "Domains (squares) along the diagonal and loop dots; the S-phase map shows the effect of replication.")
    return (fid.replace("_", " ").capitalize(), "", "")
