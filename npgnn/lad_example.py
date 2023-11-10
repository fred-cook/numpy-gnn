import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from tools import glorot_init, x_entropy, GradDescentOptim
from gcn_network import GCNNetwork

from lad_sweeper.adjacency import Adjacency
from lad_sweeper.grid_maker import GridGenerator


def draw_kkl(
    nx_G: nx.Graph, label_map, node_color, pos=None, title: str | None = None, **kwargs
):
    """Helper function to plot the network"""
    _, ax = plt.subplots(figsize=(10, 10))
    if pos is None:
        pos = nx.spring_layout(nx_G, k=5 / np.sqrt(nx_G.number_of_nodes()))

    nx.draw(
        nx_G,
        pos,
        with_labels=label_map is not None,
        labels=label_map,
        node_color=node_color,
        ax=ax,
        **kwargs
    )

    if title is not None:
        ax.set_title(title)
    plt.show()


SHAPE = (10, 10)
SIZE = SHAPE[0] * SHAPE[1]
NUM_MINES = 10

grid_gen = GridGenerator(SHAPE, NUM_MINES)
grid = grid_gen.generate_n_grids(1)[0]

# Features are the mine counts of each cell
X = np.diag(grid.flatten())
A = Adjacency(SHAPE).adjacency_matrix()
G = nx.from_numpy_array(X @ A)

D_mod = np.diag(np.sum(A, axis=1))
inv_D_mod_root = np.linalg.inv(D_mod**0.5)

A_hat = inv_D_mod_root @ A @ inv_D_mod_root

colours = {
    0: "white",
    1: "blue",
    2: "green",
    3: "red",
    4: "navy",
    5: "brown",
    6: "cyan",
    7: "black",
    8: "gray",
    -1: "orange",  # Mine
}

labels = [colours[i] for i in grid.flatten()]

print(grid.T[::-1])
draw_kkl(G, None, node_color=labels, pos=np.indices(SHAPE).flatten().reshape(2, -1).T)

gcn_model = GCNNetwork(
    n_in=SIZE,
    n_out=2,  # mine or not mine
    hidden_sizes=(16, 2),
    activation=np.tanh,
    seed=5052023,
)

## == Training ==============================================

mask_frac = 0.1
mask = int(mask_frac * SIZE)

nodes = np.arange(SIZE)
np.random.shuffle(nodes)

train_nodes = nodes[:mask]
test_nodes = nodes[mask:]

opt = GradDescentOptim(lr=2e-2, wd=2.5e-2)

embeds = []
accs = []
train_losses = []
test_losses = []

loss_min = 1e6
es_iters = 0
es_steps = 50

# for epoch in range(15000):
#     y_pred = gcn_model.forward(A_hat, X)

#     opt(y_pred, labels, train_nodes)
