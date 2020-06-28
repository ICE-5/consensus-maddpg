import numpy as np
from copy import deepcopy

def one_hot(X, bound):    
    X = np.array(X).astype(np.int8)
    n = X.size
    result = np.zeros((n, bound))
    result[np.arange(n), X] = 1

    result = result.tolist()
    return result


def add_noise(X):
    # TODO: add noise to action
    pass
    return X

