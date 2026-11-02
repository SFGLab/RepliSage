"""Toy chromatin graph: backbone, loops (LEFs) and epigenetic states (Kamada-Kawai layout)."""
import networkx as nx, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

N = 22
LOOPS = [(1, 10), (11, 17), (7, 18)]            # LEF legs (m_i, n_i)
STATES = [2,2,1,2,1,1,2,1,0,0, -1,-1,-2,-1,-2,-2,-1,0, 0,1,1,2]   # s_k in {-2..2}
COL = {2: "#C44E52", 1: "#E8A0A3", 0: "#E5E7EB", -1: "#9DBFE0", -2: "#4C78A8"}
TXT = "#1F2937"

G = nx.Graph()
G.add_nodes_from(range(N))
G.add_edges_from((k, k + 1) for k in range(N - 1))        # backbone
G.add_edges_from(LOOPS)                                   # loops
pos = nx.kamada_kawai_layout(G)

fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor="white")
nx.draw_networkx_edges(G, pos, edgelist=[(k, k + 1) for k in range(N - 1)],
                       edge_color="#6B7280", width=2.2, ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=LOOPS, edge_color="#D98E04", width=3,
                       style=(0, (4, 2)), connectionstyle="arc3,rad=0.3", arrows=True,
                       arrowstyle="-", node_size=380, ax=ax)
nx.draw_networkx_nodes(G, pos, node_color=[COL[s] for s in STATES],
                       node_size=380, edgecolors="#4B5563", linewidths=1.2, ax=ax)
nx.draw_networkx_labels(G, pos, {k: str(k) for k in range(N)}, font_size=8,
                        font_color=TXT, ax=ax)
for tag, k, off in (("$m_1$", LOOPS[0][0], (-20, 10)), ("$n_1$", LOOPS[0][1], (-22, -4))):
    ax.annotate(tag, pos[k], xytext=off, textcoords="offset points", ha="center", va="center",
                fontsize=12, color="#7A4B00")                       # legs of LEF 1
handles = [Line2D([], [], marker="o", ls="", mfc=COL[s], mec="#4B5563", ms=10, label=f"$s_k={s}$")
           for s in (2, 1, 0, -1, -2)]
handles += [Line2D([], [], color="#6B7280", lw=2.2, label="backbone"),
            Line2D([], [], color="#D98E04", lw=2.6, ls="--", label="loop (LEF)")]
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
          labelcolor=TXT, fontsize=9)
ax.set_axis_off(); ax.margins(0.08)
fig.savefig("markdown/assets/diagrams/graph_polymer.svg", bbox_inches="tight", facecolor="white")
