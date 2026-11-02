"""Graph construction and synchronization of LEF + Potts graphs."""
import networkx as nx
from collections import Counter
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from ..logger import log

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
    
    for i in log.progress(range(1,T)):
        G = create_graph(Ms[:,i], Ns[:,i], Cs[:,i])
        Ss.append(calculate_potts_synchronization(G, 5))
    
    return np.array(Ss)
