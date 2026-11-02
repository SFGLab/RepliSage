import networkx as nx
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import glob
import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm
import seaborn as sns

def magnetization(S, q=5, viz=False, out_path=None):
    """
    Computes the magnetization over time.
    Formula: M = (q * max(N_s) - N) / (N * (q - 1))
    where N_s is the count of spins in the most common state.
    ---------------------------------------------------------
    Description:
    For the Potts model, the magnetization measures how much 
    the system prefers one particular state over metadatas. 
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
    np.save(out_path+'/metadata/graph_metrics/Mag_potts.npy',M)

    if viz:
        figure(figsize=(10, 6), dpi=400)
        plt.plot(M,'g-')
        plt.xlabel('MC step',fontsize=16)
        plt.ylabel('Potts Magnetization',fontsize=16)
        if out_path!=None:
            plt.savefig(out_path+'/plots/graph_metrics/potts_model_normalized_magnetization.svg',format='svg',dpi=200)
            plt.savefig(out_path+'/plots/graph_metrics/potts_model_normalized_magnetization.png',format='png',dpi=200)
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
    np.save(out_path+'/metadata/graph_metrics/cluster_order.npy',C)

    if viz:
        figure(figsize=(10, 6), dpi=400)
        plt.plot(C,'r-')
        plt.xlabel('MC step',fontsize=16)
        plt.ylabel('Cluster Order',fontsize=16)
        if out_path!=None:
            plt.savefig(out_path+'/plots/graph_metrics/cluster_order.svg',format='svg',dpi=200)
            plt.savefig(out_path+'/plots/graph_metrics/cluster_order.png',format='png',dpi=200)
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
    np.save(out_path+'/metadata/graph_metrics/binder_cumulant.npy',U)
    
    if viz:
        figure(figsize=(10, 6), dpi=400)
        plt.plot(U,'b-')
        plt.xlabel('MC step',fontsize=16)
        plt.ylabel('Binder cumulant',fontsize=16)
        if out_path!=None:
            plt.savefig(out_path+'/plots/graph_metrics/binder_cumulant.svg',format='svg',dpi=200)
            plt.savefig(out_path+'/plots/graph_metrics/binder_cumulant.png',format='png',dpi=200)
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
    np.save(out_path+'/metadata/graph_metrics/entropy.npy',S_entropy)

    if viz:
        figure(figsize=(10, 6), dpi=400)
        plt.plot(S_entropy,'m-')
        plt.xlabel('MC step')
        plt.ylabel('S entropy')
        if out_path!=None:
            plt.savefig(out_path+'/plots/graph_metrics/entropy.svg',format='svg',dpi=200)
            plt.savefig(out_path+'/plots/graph_metrics/entropy.png',format='png',dpi=200)
        plt.grid()
        plt.close()
    return S_entropy

def overlap_order(S1, S2, out_path=None):
    """
    Computes the overlap between two configurations S1 and S2 over time.
    Formula: Q = (1/N) * sum(delta(s_i^1, s_i^2))
    """
    N, T = S1.shape
    Q = np.zeros(T)
    for t in range(T):
        Q[t] = np.mean(S1[:, t] == S2[:, t])
    np.save(out_path+'/metadata/graph_metrics/configuration_overlap.npy',Q)
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
    edges = zip(ms[ms>=0], ns[ns>=0])
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
    Calculate the synchronization metric for a Potts model, considering q states.

    Parameters:
    G (networkx.Graph): A graph where each node has a 'state' attribute.
    q (int): Number of possible states in the Potts model.

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
        total_sync += (1 if s_u == s_v else 0) / (q - 1)  # Normalize by q-1 for Potts model

    # Normalize by the number of edges
    synchronization = total_sync / num_edges if num_edges > 0 else 0
    return synchronization

def get_synch_ensemble(Ms,Ns,Cs,out_path=None):
    T = len(Ms[0,:])
    N_beads  = len(Cs)
    Ss = list()
    
    for i in tqdm(range(1,T)):
        G = create_graph(Ms[:,i], Ns[:,i], Cs[:,i])
        Ss.append(calculate_potts_synchronization(G, 5))
    
    figure(figsize=(10, 6), dpi=400)
    plt.plot(Ss,'b-')
    plt.xlabel('MC step',fontsize=16)
    plt.ylabel('Synchronization Metric',fontsize=16)
    plt.grid()
    if out_path!=None:
        plt.savefig(out_path+'/plots/graph_metrics/sync.svg',format='svg',dpi=200)
        plt.savefig(out_path+'/plots/graph_metrics/sync.png',format='png',dpi=200)
    plt.close()

def compute_potts_metrics(Ms, Ns, Cs, path):
    # Potts metrics computation
    get_synch_ensemble(Ms,Ns,Cs,path)
    magnetization(Cs+2, q=5, viz=True, out_path=path)
    cluster_order(Cs+2, viz=True, out_path=path)
    binder_cumulant(Cs+2, q=5, viz=True, out_path=path)
    entropy_order(Cs+2, q=5, viz=True, out_path=path)