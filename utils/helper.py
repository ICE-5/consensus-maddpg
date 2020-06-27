import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def noise(input):
    pass


def soft_update(target, source, tau):
	"""
	Soft update of target network (t) by absorbing weights from source network (s)
	t = tau * s + (1 - tau) * t
	:param target: target network (torch network object)
	:param source: source network (torch network object)
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)


def hard_update(target, source):
	"""
	Copy weight of source network (s) to target network (t)
	:param target: target network (torch network object)
	:param source: source network (torch network object)
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)


def one_hot(X, bound):
	X = np.array(X).astype(np.int8)
	n = X.size
	
	result = np.zeros(n, bound)
	result[np.arange(n), X] = 1

	return result


def add_noise(X):
	pass
	return X