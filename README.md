## NumPy GNN

A small example GNN written in NumPy based on [this](https://www.youtube.com/watch?v=8qTnNXdkF1Q) video and code.

Using the Zachary's Karate Club sample dataset.

Karate club has 34 nodes (people)
We will denote the number of nodes `N`. These are to be sorted into 3 categories, denoted `n`. So the input of the network has size `N` and the output has size `n`.

### Usage

Have `poetry` installed, clone the repo and run `poetry install` from within it. Enter the new venv with `poetry shell`. All of the imports are relative at the moment. Might change to a proper package one day

## Adjacency Matrix

`A` is the adjanceny matrix. It has shape `(N, N)`
`A_self = A + np.eye(len(A))` is the adjacency matrix with self connections for nodes.
`D_mod` is the sums of the rows of `A_self` placed in the diagonals of a matrix with the same shape as `A`.

`A_hat` is a normalised version of `A_self` created by:
`A_hat = D_mod^-1 @ A_mod @ D_mod^-1`

## Input features
`X` are the input features. We will start with an identity matrix with shape `(N, N)`. By doing this each node will have a column of associated values in the first layer of the GCN.

## GCN Layer

Represented by the [`GCNLayer`](npgnn/gcn_layer.py) class.

It takes a number of input features `N_in` and a number of output features `N_out` which determine the size of the learnable parameter matrix (weights) denoted `W`.

The GCN `forward` method implements the following equation:
$$H^{(l+1)}=\sigma(W\hat AH^l)$$

The node embeddings in $H^l$ (where $l$ is the layer number) are updated to $H^{l+1}$ by multiplying with the normalised adjacency matrix $\hat A$. This is where the message passing between nodes occurs. We also multiply the weights $W$ and put the output through the activation function $\sigma$.

## Softmax Layer

The final layer will be a more typical neural network layer, [`SoftmaxLayer`](npgnn/softmax_layer.py) which is very similar to the above, with the exclusion of the message passing and inclusion of some bias values. This converts the embedded values to probabilities of the different classes.

The `shift` method is to improve the numerical stability, it does not change the result.

## Gradient Descent Optimizer

There is a helper class in [`tools`](npgnn/tools.py) called `GradDescentOptim` which stores the gradients from the different layers.

It has a few attributes:
- `lr`: learning rate, the size of the steps during gradient descent
- `wd`: weight decay, modifies the weight values.
- `bs`: batch size, how many nodes included in each training step.

as well as storing the gradients from a layer to be passed to the previous layer in back propagation.

## Graph Convolutional Neural Network

The layers are stacked in a general way using [`GCNNetwork`](npgnn/gcn_network.py). This takes a number of arguments for hidden layer sizes, and puts a GCN with size `n_in` at the start, and a softmax layer which outputs `n_out` on the end.

When performing a forward pass the `GCN`s are looped over creating an embedding. These all take the adjacency matrix as a parameter. The final output from the embedding is passed into the softmax layer to be converted to a class likelihood. Note that this does not perform message passing, and therefore doesn't need `A`.

## Back Propagation

### Softmax layer

Starting with the last layer (softmax) we mask out any nodes we don't want to train on. Here we are training on them all.

`d1` stores the derivative of the loss function with respect to input values pre-softmax, which has been stored in the optimizer.
- `d1` is multiplied by the softmax layer's `W` and stored in the optimizer to be back propagated to the previous layer.
- `d1` is also multiplied with the final layers output to get the layers derivative with respect to the weight `dW`
- The partial derivative with respect to the bias `b`, `db` is also calculated.
- An additional weight decay (parameter stored in the optimizer) is calculated and added to `dW` when the weights are updated.

### GCN layers

- `dtanh` is the derivative of the $\tanh$ activation function
- The gradients from the previous layer stored in the optimizers `out` attribute are multiplied by `dtanh`
- The gradients are multiplied by the layers weights and used to update the optimizers weights.

The values from the forward pass stored in the `X` attribute already contain the message passing information, however this isn't relevant to the gradient, we're only looking for the changes from the values coming in, and the weights.