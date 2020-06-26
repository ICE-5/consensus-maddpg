import torch.nn as nn
import torch.nn.functional as F

class BaseNetwork(nn.Module):
    """
    Base network for both actor and critic
    """
    def __init__(self, input_dim, out_dim, 
                    hidden_dim=64, 
                    nonlinear=F.relu, 
                    normalize_input=True,
                    discrete_action=True):
        """
        Inputs:
            input_dim (int): imput dimensions
            out_dim (int): output dimensions
            hidden_dim (int): hidden layer dimension
            nonlinear (PyTorch function): nonlinear functions for hidden layers
        """
        super(BaseNetwork, self).__init__()

        self.nonlinear = nonlinear

        # normalize input
        if normalize_input:
            self.in_func = nn.BatchNorm1d(input_dim)
            self.in_func.weight.data.fill_(1)
            self.in_func.bias.data.fill_(0)
        else:
            self.in_func = lambda x: x

        # network structure
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

        if not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_func = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_func = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): batch of data
        Outputs:
            out (PyTorch Matrix): network output
        """
        x = self.nonlinear(self.fc1(self.in_fn(X)))
        x = self.nonlinear(self.fc2(x))
        x = self.nonlinear(self.fc3(x))
        out = self.out_func(self.out(x))
        return out