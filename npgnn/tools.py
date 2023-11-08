import numpy as np

def glorot_init(n_in: int, n_out: int):
    """
    Xavier Glorot's initialisation function
    """
    sd = np.sqrt(6.0 / (n_in + n_out))
    return np.random.uniform(-sd, sd, size=(n_in, n_out))

def x_entropy(pred: np.ndarray, labels) -> np.ndarray:
    """
    Calculate Cross entropy
    """
    return -np.log(pred)[np.arange(len(pred)),
                         np.argmax(labels, axis=1)]

def norm_diff(dW, dW_approx):
    return np.linalg.norm(dW - dW_approx) / (np.linalg.norm(dW) + np.linalg.norm(dW_approx))

class GradDescentOptim():
    """
    A helper class to store gradients between layers
    """
    def __init__(self, lr: float, wd: float):
        self.lr = lr # learning rate
        self.wd = wd # weight decay
        self._y_pred = None # output from the GCN
        self._y_true = None # True value
        self._out = None
        self.bs = None # batch size
        self.train_nodes = None

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray,
                 train_nodes: np.ndarray | None=None):
        self.y_pred = y_pred
        self.y_true = y_true

        if train_nodes is None:
            self.train_nodes = np.arange(len(y_pred))
        else:
            self.train_nodes = train_nodes

        self.bs = len(self.train_nodes)

    @property
    def out(self):
        return self._out
    
    @out.setter
    def out(self, y):
        self._out = y