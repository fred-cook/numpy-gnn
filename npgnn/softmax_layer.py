from typing import Callable

import numpy as np

from tools import GradDescentOptim

class SoftmaxLayer():
    def __init__(self, n_in: int, n_out: int,
                 weight_initialiser: Callable[[int, int], np.ndarray],
                 name: str=''):
        """
        A class for a graph convolutional network layer

        Parameters
        ----------
        n_in: int
            The size of the input values
        n_out: int
            The size of the output values
        name: str
            The name of the layer so it can be printed
            and read more easily.
        """
        self.n_inputs = n_in
        self.n_outputs = n_out
        self.W = weight_initialiser(n_out, n_in)
        self.b = np.zeros((self.n_outputs, 1)) # bias
        self.name = name

        self.X = None # Input parameters

    def __repr__(self):
        return f"Softmax: W{'_' + self.name if self.name else ''} ({self.n_inputs}, {self.n_outputs})"
    
    def shift(self, proj: np.ndarray) -> np.ndarray:
        """
        Take the exponential of the negatively shifted values
        and return them all normalised to the largest.
        """
        shiftx = proj - np.max(proj, axis=0, keepdims=True)
        exps = np.exp(shiftx)
        return exps / np.sum(exps, axis=0, keepdims=True)
    
    def forward(self, X: np.ndarray, W: np.ndarray | None=None,
                b: np.ndarray | None=None):
        """
        Compute the softmax of vector x in a numerically stable way.

        Parameters
        ----------
        X: np.ndarray
            Input values shape (batch_size, n_out)
        W: np.ndarray | None
            weights
        b: np.ndarray | None
            Bias values

        Returns
        -------

        """
        self.X = X.T
        if W is None:
            W = self.W
        if b is None:
            b = self.b

        proj = (W @ self.X) + b # (out, h)*(h, bs) = (out, bs)
        return self.shift(proj).T # (bs, out)
    
    def backward(self, optim: GradDescentOptim, update: bool=True):
        """
        Backward pass method for adjusting the weights

        Parameters
        ----------
        optim: GradDescentOptim
            An optimizer for the gradient descent
        update: bool
            whether to store the updated weights
        """
        # Any masked out nodes get a loss of 0
        train_mask = np.zeros(len(optim.y_pred))
        train_mask[optim.train_nodes] = 1
        train_mask = train_mask.reshape((-1, 1))
        
        # derivative of loss w.r.t. activation (pre-softmax)
        d1 = (optim.y_pred - optim.y_true) # (bs, out_dim)
        d1 = d1 * train_mask # (bs, out_dim) with loss of non-train nodes set to zero
        
        self.grad = d1 @ self.W # (bs, out_dim)*(out_dim, in_dim) = (bs, in_dim)
        optim.out = self.grad
        
        dW = (d1.T @ self.X.T) / optim.bs  # (out_dim, bs)*(bs, in_dim) -> (out_dim, in_dim)
        db = d1.T.sum(axis=1, keepdims=True) / optim.bs # (out_dim, 1)
                
        dW_wd = self.W * optim.wd / optim.bs # weight decay update
        
        if update:   
            self.W -= (dW + dW_wd) * optim.lr
            self.b -= db.reshape(self.b.shape) * optim.lr
        
        return dW + dW_wd, db.reshape(self.b.shape)