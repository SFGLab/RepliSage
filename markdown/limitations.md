# Known limitations

RepliSage is a deliberate simplification. Most points below are acknowledged in the paper's Discussion.

* **Fork speed is constant.** Each fork keeps one speed drawn from a normal distribution, whereas real speed varies along the genome and in time.
* **Fork-cohesin interactions are simplified.** Forks act mostly as barriers. Real encounters can be more complex: cohesive cohesin, which holds sister chromatids together, can pass through the fork, and cohesin can also be unloaded. Bypass of this kind and **fork stalling** are not modelled; unloading appears only indirectly, through $c_{\text{rep}}$ and random unbinding.
* **Replication fountains are not explained.** Replication fountains are visible in Hi-C heatmaps, but it is not clear how they would be represented in this model, since forks act mainly as barriers here and no mechanism produces such a pattern. If fountains are your focus, RepliSage is not the right tool; see [Modelling replication fountains](#modelling-replication-fountains) below.
* **Replication stress is simple.** It is slower forks plus more origins. Fork restart and the damage response are absent.
* **Abstract epigenetics.** The five states do not correspond to specific marks, and how forks reorganise them is unknown. The organiser effect is an assumption that you can switch off.
* **Pseudotime and $T_{\text{mc}}$.** Monte Carlo steps are not real time, and $T_{\text{mc}}$ has no clear biological meaning. Results depend on it.
* **Balanced, not fitted.** Coefficients are balanced by design because no data can fit them, and the model is sensitive to them.
* **Sister chromatids are symmetric.** Loop extrusion and the propagation of epigenetic states act identically on both sister chromatids: the two copies share the same LEF positions and the same states, whereas in a cell each sister has its own loops and its own pattern of marks, restored independently. A more detailed model could extrude loops and propagate states separately on each sister, and let them differ.
* **Coarse 3D model.** Beads cover 7 to 100 kb; there is no nuclear envelope, lamina or nucleolus, and mitosis is reduced to a second LEF family, topoisomerase-like relaxation and a pulling force. Transcription is not included.

## Modelling replication fountains

RepliSage is not perfect for fountains. If you want to study how DNA replication and polymer duplication reshape genome folding, including fountain-like patterns, start from the dedicated modelling work of D'Asaro and colleagues, which treats replication machineries and sister chromatids explicitly:

1. D. D'Asaro, M. M. C. Tortora, C. Vaillant, J.-M. Arbona and D. Jost, *DNA replication and polymer chain duplication reshape the genome in space and time*, Physical Review X **14**, 041020 (2024). [doi:10.1103/PhysRevX.14.041020](https://doi.org/10.1103/PhysRevX.14.041020). A minimal polymer framework with explicit replication fork progression and sister-chromatid formation; it shows that replication-dependent loop formation can shape chromosomes as loose bottle brushes during S phase.
2. D. D'Asaro, J.-M. Arbona, V. Piveteau, A. Piazza, C. Vaillant and D. Jost, *Genome-wide modeling of DNA replication in space and time confirms the emergence of replication specific patterns in vivo in eukaryotes*, Genome Biology **26**, 431 (2025). [doi:10.1186/s13059-025-03872-4](https://doi.org/10.1186/s13059-025-03872-4). A genome-wide extension compared with in vivo data.

Trust trends more than exact numbers, and treat results that depend on the organiser assumption as hypotheses. Runs are stochastic: use `--seed` and `--n-replicas`. Found something else? Please open a [GitHub issue](https://github.com/SFGLab/RepliSage/issues).
