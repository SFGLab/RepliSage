"""Co-evolution: loops (edges) and states (node colours) change together, two MC snapshots."""
import networkx as nx, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 22
COL = {2: "#C44E52", 1: "#E8A0A3", 0: "#E5E7EB", -1: "#9DBFE0", -2: "#4C78A8"}
TXT = "#1F2937"
S0 = [2,2,1,2,1,1,2,1,0,0, -1,-1,-2,-1,-2,-2,-1,0, 0,1,1,2]
S1 = S0.copy(); S1[13], S1[14], S1[15] = 1, 2, 1               # states follow the new contact
SNAP = [("Step $t$", [(1, 10), (11, 17), (7, 18)], S0),
        ("Step $t+\\Delta t$", [(1, 10), (11, 17), (5, 14)], S1)]   # one LEF moved

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor="white")
for ax, (title, loops, st) in zip(axes, SNAP):
    G = nx.path_graph(N); G.add_edges_from(loops)
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx_edges(G, pos, edgelist=list(nx.path_graph(N).edges), edge_color="#6B7280", width=2.2, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=loops, edge_color="#D98E04", width=3, style=(0, (4, 2)),
                           connectionstyle="arc3,rad=0.3", arrows=True, arrowstyle="-", node_size=300, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=[COL[s] for s in st], node_size=300,
                           edgecolors="#4B5563", linewidths=1.2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, font_color=TXT, ax=ax)
    ax.set_title(title, color=TXT, fontsize=12); ax.set_axis_off(); ax.margins(0.1)
fig.savefig("markdown/assets/diagrams/graph_coevolution.svg", bbox_inches="tight", facecolor="white")
