from typing import Callable

import numpy as np

from gcn_layer import GCNLayer
from softmax_layer import SoftmaxLayer
from tools import glorot_init

class GCNNetwork:
    def __init__(self, n_in: int, n_out: int,
                 hidden_sizes: list[int],
                 activation: Callable[np.ndarray, np.ndarray],
                 seed: int=0):
        self.n_inputs = n_in
        self.n_out = n_out
        self.hidden_sizes = hidden_sizes
        
        np.random.seed(seed)
        
        self.layers = []
        # Input layer
        gcn_in = GCNLayer(n_in, hidden_sizes[0], glorot_init, activation, name="in")
        self.layers.append(gcn_in)
        
        # Hidden layers
        for i, hidden_size in enumerate(hidden_sizes, 1):
            gcn = GCNLayer(self.layers[-1].W.shape[0],
                           hidden_size,
                            weight_initialiser=glorot_init,
                            activation=activation,
                            name=f'h{i}')
            self.layers.append(gcn)
            
        # Output layer
        sm_out = SoftmaxLayer(hidden_sizes[-1], n_out,
                              weight_initialiser=glorot_init, name='sm')
        self.layers.append(sm_out)

    def __repr__(self):
        return '\n'.join([str(l) for l in self.layers])
    
    def embedding(self, A: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Loop over all the GCN layers updating the output

        Parameters
        ----------
        A: np.ndarray
            Activation matrix (normally normalised)
        X: np.ndarray
            Input array

        Returns
        -------
        H: np.ndarray
            Output array
        """
        H = X
        for layer in self.layers[:-1]:
            H = layer.forward(A, H)
        return H
    
    def forward(self, A: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Complete forwards pass over all layers

        Parameters
        ----------
        A: np.ndarray
            Activation matrix (normally normalised)
        X: np.ndarray
            Input array

        Returns
        -------
        H: np.ndarray
            Output array
        """
        # GCN layers
        H = self.embedding(A, X)
        
        # Softmax
        p = self.layers[-1].forward(H)
        
        return np.asarray(p)