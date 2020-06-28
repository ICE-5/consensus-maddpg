import torch
import numpy as np

from collections import deque


class ReplayBuffer(object):
    def __init__(self, args, obs_dim, act_dim):
        self.buffer_size = args.buffer_size
        self.discrete_action = args.discrete_action
        self.od = obs_dim
        self.ad = act_dim

        self.num_transitions = 0
        self.buffer = deque()


    def sample_minibatch(self, batch_size):
        """Sample batch_size minibatch given current buffer content
        Args:
            batch_size (int): num of indices to sample
        Returns:
            list, int: list of sampled indices
        """        
        if self.num_transitions < batch_size:
            idxs = np.random.choice(self.num_transitions, self.num_transitions)
        else:
            idxs = np.random.choice(self.num_transitions, batch_size)
        return idxs
    

    def get_minibatch_component(self, idxs):
        # unpack transitions into categories
        component = self.buffer[idxs[0]]
        for idx in idxs[1:]:
            component = [torch.cat((item, self.buffer[idx][i]), dim=0) for i, item in enumerate(component) ]
        return component

    def size(self):
        return self.buffer_size
    
    def count(self):
        return self.num_transitions

    def add(self, curr_obs, act, next_obs, reward, done):
        # conversion to tensor
        curr_obs = torch.FloatTensor(curr_obs).flatten().unsqueeze(0)
        act      = torch.FloatTensor(act).flatten().unsqueeze(0)
        next_obs = torch.FloatTensor(next_obs).flatten().unsqueeze(0)
        reward   = torch.FloatTensor((reward, )).flatten().unsqueeze(0)
        done     = torch.FloatTensor((done, )).flatten().unsqueeze(0)

        transition = [curr_obs, act, next_obs, reward, done]        # act is in one-hot form
        if self.num_transitions < self.buffer_size:
            self.num_transitions += 1
        else:
            self.buffer.popleft()
        self.buffer.append(transition)


    def erase(self):
        self.buffer = deque()
        self.num_transitions = 0
    
