import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
# from .utils import GaussianNoise

class Actor(nn.Module):
    """
    Actor network
    """
    def __init__(self, obs_dim, act_dim, 
                    hidden_dim=64, 
                    mid_func=torch.relu,
                    out_func=torch.tanh,
                    normalize_input=False):
        """
        Inputs:
            obs_dim (int): observation dimensions
            act_dim (int): action dimensions
            hidden_dim (int): hidden layer dimension
            mid_func (PyTorch function): activation function for hidden layers
            out_func (PyTorch function): activation function for output layer
        """
        super(Actor, self).__init__()
        self.normalize = normalize_input

        self.mid_func = mid_func
        self.out_func = out_func

        # normalize input
        if normalize_input:
            self.in_func = nn.BatchNorm1d(obs_dim)
            self.in_func.weight.data.fill_(1)
            self.in_func.bias.data.fill_(0)

        # network structure
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs):
        """
        Inputs:
            obs (PyTorch Matrix): batch of observation
        Outputs:
            out (PyTorch Matrix): network output
        """
        x = self.mid_func(self.fc1(self.in_func(obs))) if self.normalize else  self.mid_func(self.fc1(obs))
        x = self.mid_func(self.fc2(x))
        out = self.out_func(self.out(x))
        return out



class Critic(nn.Module):
    """
    Critic network
    """
    def __init__(self, obs_dim, act_dim, 
                    hidden_dim=64, 
                    mid_func=torch.relu,
                    normalize_input=False):
        """
        Inputs:
            input_dim (int): imput dimensions
            out_dim (int): output dimensions
            hidden_dim (int): hidden layer dimension
            mid_func (PyTorch function): activation function for hidden layers
            out_func (PyTorch function): activation function for output layer
        """
        super(Critic, self).__init__()
        self.normalize = normalize_input

        self.mid_func = mid_func

        # normalize input
        if normalize_input:
            self.in_func = nn.BatchNorm1d(obs_dim + act_dim)
            self.in_func.weight.data.fill_(1)
            self.in_func.bias.data.fill_(0)

        # network structure
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, obs, act):
        """
        Inputs:
            X (PyTorch Matrix): batch of data
        Outputs:
            out (PyTorch Matrix): network output
        """
        X = torch.cat((obs, act), dim=-1)

        x = self.mid_func(self.fc1(self.in_func(X))) if self.normalize else self.mid_func(self.fc1(X))
        x = self.mid_func(self.fc2(x))
        out = self.out(x)
        return out