import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community.modularity_max import greedy_modularity_communities

from gcn_network import GCNNetwork
from tools import x_entropy, GradDescentOptim


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
        **kwargs,
    )

    if title is not None:
        ax.set_title(title)
    plt.show()


G = nx.karate_club_graph()
communities = greedy_modularity_communities(G)
N = G.number_of_nodes()
n = len(communities)

colours = np.array(
    sorted(
        [[idx, col] for col, com in enumerate(communities) for idx in com],
        key=lambda x: x[0],
    )
)[:, 1]


labels = np.eye(n)[colours.astype(int)]

club_labels = nx.get_node_attributes(G, "club")

A = np.where(nx.to_numpy_array(G), 1, 0)  # adjacency matrix
A_self = A + np.eye(len(A))  #  Add self connections
D_mod = np.diag(np.sum(A_self, axis=1))
inv_D_mod_root = np.linalg.inv(D_mod**0.5)

A_hat = inv_D_mod_root @ A_self @ inv_D_mod_root

X = np.eye(G.number_of_nodes())  # Input feature

gcn_model = GCNNetwork(
    n_in=N, n_out=n, hidden_sizes=(16, 2), activation=np.tanh, seed=5052023
)

_ = gcn_model.forward(A_hat, X)
embed = gcn_model.embedding(A_hat, X)

pos = {i: embed[i, :] for i in range(embed.shape[0])}
_ = draw_kkl(
    G,
    None,
    colours,
    pos=pos,
    title="Pre-training",
    cmap="gist_rainbow",
    edge_color="gray",
)


train_nodes = np.array([0, 1, 8])
test_nodes = np.array([i for i in range(labels.shape[0]) if i not in train_nodes])
opt = GradDescentOptim(lr=2e-2, wd=2.5e-2)

embeds = []
accs = []
train_losses = []
test_losses = []

loss_min = 1e6
es_iters = 0
es_steps = 50

for epoch in range(15000):
    y_pred = gcn_model.forward(A_hat, X)

    opt(y_pred, labels, train_nodes)

    for layer in reversed(gcn_model.layers):
        layer.backward(opt, update=True)

    embeds.append(gcn_model.embedding(A_hat, X))
    # Accuracy for non-training nodes
    acc = (np.argmax(y_pred, axis=1) == np.argmax(labels, axis=1))[
        [i for i in range(labels.shape[0]) if i not in train_nodes]
    ]
    accs.append(acc.mean())

    loss = x_entropy(y_pred, labels)
    loss_train = loss[train_nodes].mean()
    loss_test = loss[test_nodes].mean()

    train_losses.append(loss_train)
    test_losses.append(loss_test)

    if loss_test < loss_min:
        loss_min = loss_test
        es_iters = 0
    else:
        es_iters += 1

    if es_iters > es_steps:
        print("Early stopping!")
        break

    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch+1}, Train Loss: {loss_train:.3f}, Test Loss: {loss_test:.3f}"
        )

train_losses = np.array(train_losses)
test_losses = np.array(test_losses)

y_pred = gcn_model.forward(A_hat, X)
embed = gcn_model.embedding(A_hat, X)

pos = {i: embed[i, :] for i in range(embed.shape[0])}
_ = draw_kkl(
    G,
    None,
    colours,
    pos=pos,
    title="Pre-training",
    cmap="gist_rainbow",
    edge_color="gray",
)
