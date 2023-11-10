import numpy as np
from numpy.lib.stride_tricks import as_strided


class Adjacency:
    def __init__(self, shape: tuple[int, int]):
        self.shape = shape
        self.rows, self.columns = shape
        self.size = self.rows * self.columns
        self.neibs = self.get_neighbours()

    def get_neighbours(self) -> np.ndarray:
        """
        Return a self.shape + (3, 3) shaped array of 1D neighbour
        addresses for each cell. -1 indicates the neighbour doesn't
        exist
        """
        addresses = np.arange(self.size).reshape(*self.shape)
        padded = np.pad(addresses, 1, "constant", constant_values=(-1,))
        return as_strided(padded, self.shape + (3, 3), padded.strides * 2)

    def adjacency_matrix(self, self_connected: bool = False) -> np.ndarray:
        """
        A (self.size, self.size) shaped numpy array with 1s for
        connected nodes and 0s for unconnected nodes

        We create A with an extra column. Any non-existent neighbours
        are -1, and will be put in the last column, which is trimmed
        off when the function returns
        """
        A = np.zeros((self.size, self.size + 1))

        A[np.repeat(np.arange(self.size), 9), self.neibs.flatten()] = 1
        if not self_connected:
            A[np.diag_indices(len(A))] = 0
        return A[:, :-1]

    def edge_list(self, self_connected: bool = False) -> np.ndarray:
        """
        return a numpy array of edge pairs
        """
        counts = np.sum(self.neibs != -1, axis=(-1, -2)).flatten()
        pairs = np.c_[
            np.repeat(np.arange(self.size), counts),
            self.neibs[self.neibs != -1].flatten(),
        ]
        if not self_connected:
            pairs = pairs[(pairs[:, 0] - pairs[:, 1]) != 0]
        return pairs.T
