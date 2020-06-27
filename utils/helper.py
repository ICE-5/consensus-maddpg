import numpy as np
from copy import deepcopy

def one_hot(X, bound):
        # X = np.array(X).astype(np.int8)
	# n = X.size

        b = len(X)
        tmp = np.zeros(bound)
        result = [deepcopy(tmp)[X[i]]=1 for i in ramge(n)]
	
	#result = np.zeros((n, bound))
	# result[np.arange(n), X] = 1

	return result


def add_noise(X):
	# TODO: add noise to action
	pass
	return X
