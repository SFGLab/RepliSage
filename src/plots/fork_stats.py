"""LEF-fork encounters in S phase: cumulative course, counts and rates."""
import numpy as np
import matplotlib.pyplot as plt
from .style import CYCLE, GREY, save_fig
from ..analysis.fork_stats import fork_summary

SUB = "replication"

def fig_fork_events(res, p, ph, out):
    """Cumulative fraction of encounters / pushes / removals, their counts per S-phase tenth, and the rates."""
    s = fork_summary(res, p, ph)
    if s is None: return
    c_enc, c_push, c_rem = CYCLE[0], CYCLE[1], CYCLE[2]; b = s["bins"]; w = .04
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.4), gridspec_kw=dict(wspace=.3))
    ax = axs[0]; ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1, label="uniform in time")
    for k, c, name in (("enc", c_enc, "encounters"), ("push", c_push, "pushed"), ("rem", c_rem, "removed")): ax.plot(s["x"], s["cum"][k], color=c, lw=2, label=name)
    ax.set(xlabel="S-phase progress", ylabel="Cumulative fraction of events", xlim=(0, 1), ylim=(0, 1.02), title="Cumulative fraction over S phase"); ax.legend(loc="upper left")
    ax = axs[1]
    ax.bar(b["mid"] - w / 2, b["push"], w, color=c_push, label="pushed", edgecolor="white")
    ax.bar(b["mid"] + w / 2, b["rem"], w, color=c_rem, label="removed", edgecolor="white")
    ax.set(xlabel="S-phase progress", ylabel="Events per tenth of S phase", title=f"Events: {int(s['push'])} pushed, {int(s['rem'])} removed"); ax.legend()
    ax = axs[2]
    ax.plot(b["mid"], b["push_rate"], "-o", color=c_push, ms=4, lw=1.8, label=f"pushing rate (mean {s['push_rate']:.2f})")
    ax.plot(b["mid"], b["rem_rate"], "-o", color=c_rem, ms=4, lw=1.8, label=f"removal rate (mean {s['rem_rate']:.2f})")
    ax.set(xlabel="S-phase progress", ylabel="Events per encounter", xlim=(0, 1), ylim=(0, None), title="Rates per LEF-fork encounter"); ax.legend()
    return save_fig(fig, out, SUB, "fork_events", "fork_events")
