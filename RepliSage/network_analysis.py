import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm
import seaborn as sns

def plot_probability_distro(Ls, out_path=None):
    """
    Plots how the probability distribution changes over columns of matrix Ls using plt.imshow.
    
    Parameters:
        Ls (np.ndarray): 2D array where rows represent samples, and columns represent time points.
        out_path (str, optional): Path to save the heatmap. If None, it will only display the plot.
    """
    # Normalize Ls along rows to get probabilities
    Ls = np.log(Ls)
    min_val, max_val = np.min(Ls), np.max(Ls)
    bin_edges = np.linspace(min_val, max_val, 20)  # Adjust number of bins for desired resolution
    
    # Compute histogram for each column
    hist_matrix = []
    for col in Ls.T:
        hist, _ = np.histogram(col, bins=bin_edges, density=True)
        hist_matrix.append(hist)
    
    hist_matrix = np.array(hist_matrix).T  # Shape: (bins, time points)
    
    # Create the heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(hist_matrix, aspect='auto', cmap='rainbow', origin='lower', interpolation='bicubic', vmax=np.average(Ls)/2,
               extent=[0, Ls.shape[1], bin_edges[0], bin_edges[-1]])
    
    # Add colorbar
    plt.colorbar(label="Probability Density")
    
    # Labels and title
    plt.title("Probability Distribution Over Time", fontsize=18)
    plt.xlabel("Time (Columns of Ls)", fontsize=16)
    plt.ylabel("Logarithm of Loop Length", fontsize=16)
    
    # Optionally save the plot
    if out_path:
        plt.savefig(out_path + '/probability_distro_heatmap.png', format='png', dpi=200)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def loop_distro(Ls, rep_start_t, rep_end_t, out_path=None):    
    # Prepare data for violin plots
    before_replication = np.log(Ls[:, :rep_start_t].flatten())
    during_replication = np.log(Ls[:, rep_start_t:rep_end_t].flatten())
    after_replication = np.log(Ls[:, rep_end_t:].flatten())

    # # Remove outliers
    # before_replication = before_replication[before_replication < 50]
    # during_replication = during_replication[during_replication < 50]
    # after_replication = after_replication[after_replication < 50]

    # Combine data into a single structure for seaborn
    data = np.concatenate([before_replication, during_replication, after_replication])
    labels = (['Before Replication'] * len(before_replication) +
              ['During Replication'] * len(during_replication) +
              ['After Replication'] * len(after_replication))

    # Create a violin plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(x=labels, y=data, palette='muted', inner="box", linewidth=2, bw=0.5)  # Adjust box width and smoothing

    # Add title and labels
    plt.title('Regular DNA Replication', fontsize=18)
    plt.ylabel('Log Loop Length', fontsize=16)
    plt.xlabel('Replication Stage', fontsize=16)

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Optionally save the plot
    if out_path:
        plt.savefig(out_path + '/loop_dist.svg', format='svg', dpi=200)

def magnetization(S, q=5, viz=False, out_path=None):
    """
    Computes the magnetization over time.
    Formula: M = (q * max(N_s) - N) / (N * (q - 1))
    where N_s is the count of spins in the most common state.
    ---------------------------------------------------------
    Description:
    For the Potts model, the magnetization measures how much 
    the system prefers one particular state over others. 
    This order parameter ranges from:
    
    M=0: Symmetric (disordered) phase (e.g., in the high-temperature regime),
    M=1: Fully ordered phase where all spins align in one state (e.g., in the low-temperature regime).
    """
    N, T = S.shape
    M = np.zeros(T)
    for t in range(T):
        state_counts = np.bincount(S[:, t], minlength=q)
        max_state = np.max(state_counts)
        M[t] = (q * max_state - N) / (N * (q - 1))

    if viz:
        plt.plot(M,'go-')
        plt.xlabel('MC step',fontsize=16)
        plt.ylabel('Potts Magnetization',fontsize=16)
        if out_path!=None:
            plt.savefig(out_path+'/potts_model_normalized_magnetization.svg',format='svg',dpi=200)
        plt.grid()
        plt.close()
    return M

def cluster_order(S, viz=False, out_path=None):
    """
    Computes the cluster-based order parameter over time.
    Formula: C = <S_max> / N, where S_max is the size of the largest cluster.
    -------------------------------------------------------------------------
    Description:
    In the disordered phase, clusters are small, and C is close to 0. In the ordered phase,
    the largest cluster spans the system, and C approaches 1.

    Note:
    Here we refer as clusters how they form linearly without taking ito account the interaction matrix.
    """
    N, T = S.shape
    C = np.zeros(T)
    for t in range(T):
        state_counts = np.bincount(S[:, t])
        largest_cluster = np.max(state_counts)
        C[t] = largest_cluster / N

    if viz:
        plt.plot(C,'ro-')
        plt.xlabel('MC step',fontsize=16)
        plt.ylabel('Cluster Order',fontsize=16)
        if out_path!=None:
            plt.savefig(out_path+'/cluster_order.svg',format='svg',dpi=200)
        plt.grid()
        plt.close()
    return C

def binder_cumulant(S, q=5, viz=False, out_path=None):
    """
    Computes the Binder cumulant over time.
    Formula: U = 1 - <M^4> / (3 * <M^2>^2)
    """
    N, T = S.shape
    U = np.zeros(T)
    for t in range(T):
        state_counts = np.bincount(S[:, t], minlength=q)
        probs = state_counts / N
        m2 = np.sum(probs**2)
        m4 = np.sum(probs**4)
        U[t] = 1 - m4 / (3 * m2**2)
    
    if viz:
        plt.plot(U,'bo-')
        plt.xlabel('MC step',fontsize=16)
        plt.ylabel('Binder cumulant',fontsize=16)
        if out_path!=None:
            plt.savefig(out_path+'/binder_cumulant.svg',format='svg',dpi=200)
        plt.grid()
        plt.close()
    return U

def entropy_order(S, q=5, viz=False,out_path=None):
    """
    Computes the entropy over time.
    Formula: S = -sum(P_s * log(P_s)) where P_s is the probability of state s.
    -------------------------------------------------------------------------
    In the ordered phase:
    
    Ps​≈1 for one dominant state, and S→0S→0.

    In the disordered phase:

    Ps​≈1/q, and S→ln⁡(q)S→ln(q).
    """
    N, T = S.shape
    S_entropy = np.zeros(T)
    for t in range(T):
        state_counts = np.bincount(S[:, t], minlength=q)
        probs = state_counts / N
        S_entropy[t] = entropy(probs, base=np.e)

    if viz:
        plt.plot(S_entropy,'mo-')
        plt.xlabel('MC step')
        plt.ylabel('S entropy')
        if out_path!=None:
            plt.savefig(out_path+'/entropy.svg',format='svg',dpi=200)
        plt.grid()
        plt.close()
    return S_entropy

def overlap_order(S1, S2):
    """
    Computes the overlap between two configurations S1 and S2 over time.
    Formula: Q = (1/N) * sum(delta(s_i^1, s_i^2))
    """
    N, T = S1.shape
    Q = np.zeros(T)
    for t in range(T):
        Q[t] = np.mean(S1[:, t] == S2[:, t])
    return Q

def visualize_potts_graph(G):
    """
    Visualize a graph with nodes having 5 possible states: -2, -1, 0, 1, 2.

    Parameters:
    G (networkx.Graph): A graph where each node has a 'state' attribute.

    Returns:
    None
    """
    # Define node colors based on state
    color_map = {
        -2: 'purple',
        -1: 'blue',
        0: 'gray',
        1: 'orange',
        2: 'red'
    }

    # Check if each node has the 'state' attribute, and use a default value if not
    node_color = []
    for node in G.nodes:
        state = G.nodes[node].get('state', 0)  # Default to 0 if 'state' is missing
        node_color.append(color_map.get(state, 'gray'))  # Default to gray for unknown states

    # Choose a layout
    pos = nx.kamada_kawai_layout(G)  # Using Kamada-Kawai layout for a natural appearance

    # Draw the graph
    nx.draw(G, pos, with_labels=False, node_color=node_color, node_size=1,
            edge_color='black', width=1)  # Highlight edges with increased width
    plt.close()

def create_graph(ms, ns, cs):
    """
    Create a NetworkX graph from node states and link indices.

    Parameters:
    ms (list or array): List of source node indices for edges.
    ns (list or array): List of target node indices for edges.
    cs (list or array): List of node states corresponding to node indices.

    Returns:
    networkx.Graph: A graph with nodes and edges, where each node has a 'state' attribute.
    """
    # Create an empty graph
    G = nx.Graph()

    # Add nodes with states
    for i, state in enumerate(cs):
        G.add_node(i, state=state)

    # Add edges
    edges = zip(ms, ns)
    G.add_edges_from(edges)

    # Connect consecutive nodes by their index
    for i in range(len(cs) - 1):  # Using length of cs to connect consecutive nodes
        G.add_edge(i, i + 1)

    return G

def calculate_ising_synchronization(G):
    """
    Calculate the synchronization metric of a graph based on node states.

    Parameters:
    G (networkx.Graph): A graph where each node has a 'state' attribute (-1 or 1).

    Returns:
    float: Synchronization metric (0 to 1).
    """
    if not nx.get_node_attributes(G, 'state'):
        raise ValueError("Graph nodes must have a 'state' attribute assigned.")

    total_sync = 0
    num_edges = G.number_of_edges()

    for u, v in G.edges:
        s_u = G.nodes[u]['state']
        s_v = G.nodes[v]['state']
        total_sync += (1 + s_u * s_v) / 2

    # Normalize by the number of edges
    synchronization = total_sync / num_edges if num_edges > 0 else 0
    return synchronization

def calculate_potts_synchronization(G, q):
    """
    Calculate the synchronization metric for a Potts model with q states.

    Parameters:
    G (networkx.Graph): A graph where each node has a 'state' attribute (1 to q).
    q (int): The number of states in the Potts model.

    Returns:
    float: Synchronization metric (0 to 1).
    """
    if not nx.get_node_attributes(G, 'state'):
        raise ValueError("Graph nodes must have a 'state' attribute assigned.")

    total_sync = 0
    num_edges = G.number_of_edges()

    for u, v in G.edges:
        s_u = G.nodes[u]['state']
        s_v = G.nodes[v]['state']
        total_sync += int(s_u == s_v)  # Add 1 if states are the same, 0 otherwise

    # Normalize by the number of edges
    synchronization = total_sync / num_edges if num_edges > 0 else 0
    return synchronization

def get_synch_ensemble(Ms,Ns,Cs,out_path=None):
    T = len(Ms[0,:])
    N_beads  = len(Cs)
    Ss = list()

    for i in tqdm(range(1,T)):
        G = create_graph(Ms[:N_beads,i], Ns[:N_beads,i], Cs[:N_beads,i])
        Ss.append(calculate_potts_synchronization(G, 5))

    plt.plot(Ss,'bo-')
    plt.xlabel('MC step',fontsize=16)
    plt.ylabel('Synchronization Metric',fontsize=16)
    plt.grid()
    if out_path!=None:
        plt.savefig(out_path+'/sync.pdf',dpi=200)
    plt.close()

def compute_potts_metrics(N_beads,in_path,out_path):
    # Import data
    Cs = np.load(in_path+'/spin_traj.npy')
    Ms = np.load(in_path+'/Ms.npy')
    Ns = np.load(in_path+'/Ns.npy')

    # Potts metrics computation
    G = create_graph(Ms[:N_beads,0], Ns[:N_beads,0], Cs[:N_beads,0])
    # visualize_potts_graph(G)
    get_synch_ensemble(Ms,Ns,Cs,out_path)
    magnetization(Cs[:,1:]+2, q=5, viz=True, out_path=out_path)
    cluster_order(Cs[:,1:]+2, viz=True, out_path=out_path)
    binder_cumulant(Cs[:,1:]+2, q=5, viz=True, out_path=out_path)
    entropy_order(Cs[:,1:]+2, q=5, viz=True, out_path=out_path)

def run():
    Ms = np.load('/home/skorsak/Projects/mine/RepliSage/stress_test_region/other/Ms.npy')
    Ns = np.load('/home/skorsak/Projects/mine/RepliSage/stress_test_region/other/Ns.npy')
    Ls = Ns-Ms
    loop_distro(Ls, 60, 100, out_path=None)