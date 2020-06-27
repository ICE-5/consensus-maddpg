import numpy as np
from copy import deepcopy

def one_hot(X, bound):
    
    print(len(X))
    X = np.array(X).astype(np.int8)
    print(f'here: {X.shape}')
    n = X.size
    result = np.zeros((n, bound))
    result[np.arange(n), X] = 1

    return result


def add_noise(X):
    # TODO: add noise to action
    pass
    return X
