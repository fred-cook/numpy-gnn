import numpy as np

def glorot_init(n_in: int, n_out: int):
    """
    Xavier Glorot's initialisation function
    """
    sd = np.sqrt(6.0 / (n_in + n_out))
    return np.random.uniform(-sd, sd, size=(n_in, n_out))

def x_entropy(pred: np.ndarray, labels):
    """
    Calculate Cross entropy
    """
    return -np.log(pred)[np.arange(len(pred)),
                         np.argmax(labels, axis=1)]

def norm_diff(dW, dW_approx):
    return np.linalg.norm(dW - dW_approx) / (np.linalg.norm(dW) + np.linalg.norm(dW_approx))

class GradDescentOptim():
    def __init__(self, lr, wd):
        self.lr = lr
        self.wd = wd
        self._y_pred = None
        self._y_true = None
        self._out = None
        self.bs = None
        self.train_nodes = None

    def __call__(self, y_pred, y_true, train_nodes=None):
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