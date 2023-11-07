from typing import Callable

import numpy as np

from tools import GradDescentOptim

class GCNLayer:
    def __init__(self, n_in: int, n_out: int,
                 weight_initialiser: Callable([int, int], np.ndarray),
                 activation: Callable(np.ndarray, np.ndarray) | None=None,
                 name: str=''):
        """
        A class for a graph convolutional network layer

        Parameters
        ----------
        n_in: int
            The size of the input values
        n_out: int
            The size of the output values
        activation: Callable(np.ndarray, np.ndarray)
            An activation function eg. sigmoid / reLU
        name: str
            The name of the layer so it can be printed
            and read more easily.
        """
        self.n_in = n_in
        self.n_out = n_out
        self.W = weight_initialiser(self.n_out, self.n_in) # weights
        self.activation = activation
        self.name = name

        self.A: np.ndarray # Activation layer shape (batch_size, batch_size)
        self.X: np.ndarray # Network shape (batch_size, input_feature_length)
        self.H: np.ndarray # Outputs from forward pass
        self.grad: np.ndarray

    def __repr__(self):
        return f"GCN: W{'_'+self.name if self.name else ''} ({self.n_in}, {self.n_out})"
    
    def forward(self, A: np.ndarray, X: np.ndarray,
                W: np.ndarray | None=None) -> np.ndarray:
        """
        A method for a forward pass through the network

        Parameters
        ----------
        A: np.ndarray
            The activation matrix
        X: np.ndarray
            The feature matrix
        W: np.ndarray | None
            The current state of the weights. On the first
            pass None

        Returns
        -------
        H: np.ndarray
            output array
        """
        self.A = A
        self.X = (A @ X).T # message passing

        if W is None:
            W = self.W

        H = W @ self.X # (n_inputs, n_outputs)(n_outputs, batch_size) -> (n_inputs, batch_size)
        if self.activation is not None:
            H = self.activation(H)

        self.H = H
        return H.T
    
    def backward(self, optim: GradDescentOptim, update: bool=True) -> np.ndarray:
        """
        Backward pass method for adjusting the weights

        Parameters
        ----------
        optim: GradDescentOptim
            An optimizer for the gradient descent
        update: bool
            whether to store the updated weights
        """
        dtanh = 1 - self.H.T**2
        d2 = optim.out * dtanh  # (bs, out_dim) *element_wise* (bs, out_dim)
        
        self.grad = self.A @ d2 @ self.W # (bs, bs)*(bs, out_dim)*(out_dim, in_dim) = (bs, in_dim)     
        optim.out = self.grad
        
        dW = np.asarray(d2.T @ self.X.T) / optim.bs  # (out_dim, bs)*(bs, D) -> (out_dim, D)
        dW_wd = self.W * optim.wd / optim.bs # weight decay update
        
        if update:
            self.W -= (dW + dW_wd) * optim.lr 
        
        return dW + dW_wd