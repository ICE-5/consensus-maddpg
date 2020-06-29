import numpy as np
from copy import deepcopy

def one_hot(X, bound):    
    X = np.array(X).astype(np.int8)
    n = X.size
    result = np.zeros((n, bound))
    result[np.arange(n), X] = 1

    result = result.tolist()
    return result

def one_hot_random_action(bound):
    """
    :This function only generate an action for ONE agent.
    :type bound: action dimension
    :rtype: one hot action, 1d list
    """
    p = np.random.rand(bound)
    idx = np.argmax(p)
    a = np.zeros(bound)
    a[idx] = 1.
    return a.tolist()


def add_noise(X, noise_rate):
    """
    : Each agent has probability noise_rate to generate a new one hot action.
    """
    prob_arr = np.random.rand(len(X))
    idx_n = prob_arr < noise_rate
    if idx_n.size == 0:
        return X

    idx_n = idx_n.tolist()
    X = [one_hot_random_action(len(X[0])) if b else X[i] for i, b in enumerate(idx_n)]
    return X

