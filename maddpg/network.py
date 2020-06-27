import torch.nn as nn
import torch.nn.functional as F

class BaseNetwork(nn.Module):
    """
    Base network for both actor and critic
    """
    def __init__(self, input_dim, out_dim, 
                    hidden_dim=64, 
                    mid_func=F.relu,
                    out_func=lambda x: x, 
                    normalize_input=True):
        """
        Inputs:
            input_dim (int): imput dimensions
            out_dim (int): output dimensions
            hidden_dim (int): hidden layer dimension
            mid_func (PyTorch function): activation function for hidden layers
            out_func (PyTorch function): activation function for output layer
        """
        super(BaseNetwork, self).__init__()

        self.mid_func = mid_func

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

        # TODO: add re-weighting / normalization tricks for neural net 

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): batch of data
        Outputs:
            out (PyTorch Matrix): network output
        """
        x = self.mid_func(self.fc1(self.in_func(X)))
        x = self.mid_func(self.fc2(x))
        x = self.mid_func(self.fc3(x))
        out = self.out_func(self.out(x))
        return out
