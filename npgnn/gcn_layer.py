import numpy as np

from tools import glorot_init
from tools import GradDescentOptim

class GCNLayer:
    def __init__(self, n_inputs, n_outputs, activation=None, name=''):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.W = glorot_init(self.n_outputs, self.n_inputs)
        self.activation = activation
        self.name = name

        self.A: np.ndarray # Activation layer size (batch_size, batch_size)
        self.X: np.ndarray # Network shape (batch_size, input_feature_length)
        self.H: np.ndarray # Outputs from forward pass
        self.grad: np.ndarray

    def __repr__(self):
        return f"GCN: W{'_'+self.name if self.name else ''} ({self.n_inputs}, {self.n_outputs})"
    
    def forward(self, A, X, W=None):
        self.A = A
        self.X = (A @ X).T

        if W is None:
            W = self.W

        H = W @ self.X # (n_inputs, n_outputs)(n_outputs, batch_size) -> (n_inputs, batch_size)
        if self.activation is not None:
            H = self.activation(H)

        self.H = H
        return H.T
    
    def backward(self, optim: GradDescentOptim, update=True):
        dtanh = 1 - self.H.T**2
        d2 = optim.out * dtanh  # (bs, out_dim) *element_wise* (bs, out_dim)
        
        self.grad = self.A @ d2 @ self.W # (bs, bs)*(bs, out_dim)*(out_dim, in_dim) = (bs, in_dim)     
        optim.out = self.grad
        
        dW = np.asarray(d2.T @ self.X.T) / optim.bs  # (out_dim, bs)*(bs, D) -> (out_dim, D)
        dW_wd = self.W * optim.wd / optim.bs # weight decay update
        
        if update:
            self.W -= (dW + dW_wd) * optim.lr 
        
        return dW + dW_wd