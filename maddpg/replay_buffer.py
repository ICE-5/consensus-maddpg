from collections import deque
import random
import pickle


class ReplayBuffer(object):
    def __init__(self, args):
        self.buffer_size = args.buffer_size
        self.num_transitions = 0
        self.discrete_action = args.discrete_action
        self.n = args.num_agents
        self.od = args.obs_dim
        self.ad = args.action_dim

        self.buffer = deque()

        self.curr_obs_ids = 0                                       # start idx of curr_obs_n in transition
        self.curr_obs_ide = self.n * self.od                        # end idx of curr_obs_n in transition
        self.next_obs_ids = self.curr_obs_ide
        self.next_obs_ide = self.next_obs_ids + self.n * self.od
        self.action_ids = self.next_obs_ide
        self.action_ide = self.action_ids + self.n * self.ad
        self.reward_ids = self.action_ide
        self.reward_ide = self.reward_ids + self.n
        self.done_ids = self.reward_ide
        self.done_ide = self.done_ids + self.n

    def sample_minibatch(self, batch_size):
        """Sample batch_size minibatch given current buffer content
        Args:
            batch_size (int): num of indices to sample
        Returns:
            list, int: list of sampled indices
        """        
        if self.num_transitions < batch_size:
            minibatch = random.sample(self.buffer, self.num_transitions)
        else:
            minibatch = random.sample(self.buffer, batch_size)
        
        minibatch = torch.tensor(minibatch, dtype=torch.float32)
        curr_obs_n = minibatch[:, self.curr_obs_ids : self.curr_obs_ide]
        next_obs_n = minibatch[:, self.next_obs_ids : self.next_obs_ide]
        action_n = minibatch[:, self.action_ids : self.action_ide]
        reward_n = minibatch[:, self.reward_ids : self.reward_ide]
        done_n = minibatch[:, self.done_ids : self.done_ide]

        return curr_obs_n, next_obs_n, action_n, reward_n, done_n
    

    def get_batch(self, idxs):
        return self.buffer[idxs]

    def size(self):
        return self.buffer_size
    
    def count(self):
        return self.num_transitions

    def add(self, curr_obs_n, next_obs_n, action_n, reward_n, done_n):
        curr_obs_n = torch.tensor(curr_obs_n, dtype=torch.float32).flatten()
        next_obs_n = torch.tensor(next_obs_n, dtype=torch.float32).flatten()

        if self.discrete_action:
            action_n = torch.flatten(action_n)
        else:
            action_n = torch.tensor(action_n, dtype=torch.float32).flatten()

        reward_n = torch.tensor(reward_n, dtype=torch.float32).flatten()
        done_n = torch.tensor(reward_n, dtype=torch.float32).flatten()

        transition = torch.cat((curr_obs_n, next_obs_n, action_n, reward_n, done_n))

        if self.num_transitions < self.buffer_size:
            self.num_transitions += 1
        else:
            self.buffer.popleft()

        self.buffer.append(transition)


    def erase(self):
        self.buffer = deque()
        self.num_transitions = 0
    