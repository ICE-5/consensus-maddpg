from collections import deque
import random
import pickle

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_transitions = 0
        self.buffer = deque()
        
    def get_batch_idx(self, batch_size):
        """Sample batch_size indices given current buffer content
        Args:
            batch_size (int): num of indices to sample
        Returns:
            list, int: list of sampled indices
        """        
        if self.num_transitions < batch_size:
            return random.sample(self.num_transitions, self.num_transitions)
        else:
            return random.sample(self.num_transitions, batch_size)
    
    def get_selected_sample(self, idxs):
        """Retrieve sample from buffer using given indices
        Args:
            idxs (list, int): list of indices
        Returns:
            list, xx: list of samples retrieved from buffer
        """        
        return self.buffer[idxs]

    def size(self):
        return self.buffer_size
    
    def count(self):
        return self.num_transitions

    def add(self, state, action, reward, new_state, done):
        transition = (state, action, reward, new_state, done)
        if self.num_transitions < self.buffer_size:
            self.buffer.append(transition)
            self.num_transitions += 1
        else:
            self.buffer.popleft()
            self.buffer.append(transition)

    def erase(self):
        self.buffer = deque()
        self.num_transitions = 0
    