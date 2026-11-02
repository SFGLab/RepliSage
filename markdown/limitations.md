# Known limitations

RepliSage is a deliberate simplification. Most points below are acknowledged in the paper's Discussion.

* **Fork speed is constant.** Each fork keeps one speed drawn from a normal distribution, whereas real speed varies along the genome and in time.
* **Forks are mostly barriers.** Bypass or cohesin unloading appear only indirectly, through $c_{\text{rep}}$ and random unbinding. Sister chromatid cohesion and cohesive cohesin are not modelled.
* **Replication stress is simple.** It is slower forks plus more origins. Stalling, restart and the damage response are absent.
* **Abstract epigenetics.** The five states do not correspond to specific marks, and how forks reorganise them is unknown. The organiser effect is an assumption that you can switch off.
* **Pseudotime and $T_{\text{mc}}$.** Monte Carlo steps are not real time, and $T_{\text{mc}}$ has no clear biological meaning. Results depend on it.
* **Balanced, not fitted.** Coefficients are balanced by design because no data can fit them, and the model is sensitive to them.
* **Coarse 3D model.** Beads cover 7 to 100 kb; there is no nuclear envelope, lamina or nucleolus, and mitosis is reduced to a second LEF family, topoisomerase-like relaxation and a pulling force. Transcription is not included.
* **Limited validation.** There are no reference 3D structures, so validation relies on Hi-C compartments, single-cell Hi-C trends and HiChIP under mild stress in one cell line. Timing for K562 is borrowed from GM12878.

Trust trends more than exact numbers, and treat results that depend on the organiser assumption as hypotheses. Runs are stochastic: use `--seed` and `--n-replicas`. Found something else? Please open a [GitHub issue](https://github.com/SFGLab/RepliSage/issues).
