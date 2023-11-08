## NumPy GNN

A small example GNN written in NumPy based on [this](https://www.youtube.com/watch?v=8qTnNXdkF1Q) video and code.

Using the Zachary's Karate Club sample dataset.

Karate club has 34 nodes (people)
We will denote the number of nodes `N`.

## Adjacency Matrix

`A` is the adjanceny matrix. It has shape `(N, N)`
'A_self = A + np.eye(len(A))` is the adjacency matrix with self connections for nodes.
`D_mod` is the sums of the rows of `A_self` placed in the diagonals of a matrix with the same shape as `A`.

`A_hat` is a normalised version of `A_self` created by:
`A_hat = D_mod^-1 @ A_mod @ D_mod^-1`

## Input features
`X` are the input features. We will start with an identity matrix with shape `(N, N)`. By doing this each node will have a column of associated values in the first layer of the GCN.

## GCN Layer

Represented by the `GCNLayer` class.

It takes a number of input features `N_in` and a number of output features `N_out` which determine the size of the learnable parameter matrix (weights) denoted `W`.

The GCN `forward` method implements the following equation:
$$ H^(l + 1) = \sigma(WAH^l) $$

