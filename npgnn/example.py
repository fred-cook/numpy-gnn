import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community.modularity_max import greedy_modularity_communities

from gcn_layer import GCNLayer
from softmax_layer import SoftmaxLayer
from tools import GradDescentOptim

def draw_kkl(nx_G: nx.Graph, label_map, node_color, pos=None, **kwargs):
    """Helper function to plot the network"""
    _, ax = plt.subplots(figsize=(10,10))
    if pos is None:
        pos = nx.spring_layout(nx_G, k=5/np.sqrt(nx_G.number_of_nodes()))

    nx.draw(
        nx_G, pos, with_labels=label_map is not None, 
        labels=label_map, 
        node_color=node_color, 
        ax=ax, **kwargs)
    
G = nx.karate_club_graph()
communities = greedy_modularity_communities(G)
colours = np.zeros(G.number_of_nodes())
for i, com in enumerate(communities):
    colours[list(com)] = i

n_classes = np.unique(colours).shape[0]
labels = np.eye(n_classes)[colours.astype(int)]

club_labels = nx.get_node_attributes(G,'club')
draw_kkl(G, None, colours, cmap='gist_rainbow', edge_color='gray')

A = np.where(nx.to_numpy_array(G), 1, 0) # adjacency matrix
A_self = A + np.eye(len(A)) #  Add self connections
D_mod = np.diag(np.sum(A_self, axis=1))
inv_D_mod_root = np.linalg.inv(D_mod**0.5)

A_hat = inv_D_mod_root @ A_self @ inv_D_mod_root

X = np.eye(G.number_of_nodes()) # Input features
