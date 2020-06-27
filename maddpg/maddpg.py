import torch
import os

from agent import Agent
from replay_buffer import ReplayBuffer
from utils.helper import *


class MADDPG:
    def __init__(self, args, env):    
        self.n = args.num_agent         # num of agents
        self.ad = args.action_dim       # action dim
        self.od = args.obs_dim          # obs dim
        self.batch_size = args.batch_size

        self.gamma = args.gamma
        self.env = env
        self.agents = [Agent(args) for i in range(self.n)]
        self.buffer = ReplayBuffer(args)
        self.args = args
    
    def memory_burnin(self):
        # TODO: add burn-in process to buffer
        pass


    def agent_critic_loss(self, agent_idx, curr_obs_n, next_obs_n, action_n, reward_n):
        # calculate y
        o = next_obs_n
        a = torch.empty((self.batch_size, self.n*self.ad), dtype=torch.float32)
        for i, agent in enumerate(self.agents):
            curr_obs = curr_obs_n[:, i*self.ad : (i+1)*self.ad]
            action = agent.get_action(curr_obs, is_target=True, explore=False)
            a[:, i*self.ad : (i+1)*self.ad] = action
        y = torch.cat([o, a])
        with torch.no_grad():
            y = self.agents[agent_idx].get_q(y, is_target=True, explore=False)
        y = self.gamma * y + reward_n[:, agent_idx]

        # calculate goal_y
        goal_o = curr_obs_n
        goal_a = action_n
        goal_y = torch.cat([goal_o, goal_a])
        with torch.no_grad():
            goal_y = self.agents[agent_idx].get_q(y, is_target=False, explore=False)
        
        # square loss
        return (y - goal_y).pow(2).mean()
    

    def agent_actor_loss(self, agent_idx, curr_obs_n, action_n):
        o = curr_obs_n
        a = action_n

        with torch.no_grad():
            agent_curr_obs = curr_obs_n[:, agent_idx*self.od : (agent_idx+1)*self.od]
            agent_action = self.agents[agent_idx].get_action(agent_curr_obs, is_target=False, explore=False, decode=False)
            a[:, agent_idx*self.ad : (agent_idx+1)*self.ad] = agent_action

            loss = - self.agents[agent_idx].critic(torch.cat([o, a])).mean()
        return loss

    
    def agent_update(self, agent_idx):
        curr_obs_n, next_obs_n, action_n, reward_n, _ = self.buffer.sample_minibatch(self.batch_size)

        self.agents[agent_idx].critic_optim.zero_grad()
        self.agent_critic_loss(agent_idx, curr_obs_n, next_obs_n, action_n, reward_n).backward()
        self.agents[agent_idx].critic_optim.step()
        
        self.agents[agent_idx].actor_optim.zero_grad()
        self.agent_actor_loss(agent_idx, curr_obs_n, action_n).backward()
        self.agents[agent_idx].actor_optim.step()
    

    def agent_target_update(self, agent_idx, tau):
        self.agents[agent_idx].target_update(self.args.tau, actor=True)
        self.agents[agent_idx].target_update(self.args.tau, actor=False)


    def train(self):
        for episode in range(self.args.num_episodes):
            step = 0
            curr_obs_n = self.env.reset()
            while True:
                action_n = [agent.get_action(curr_obs_n[i]) for i, agent in enumerate(self.agents) ]
                next_obs_n, reward_n, done_n, _ = self.env.step(action_n)

                action_n = one_hot(action_n, self.args.action_dim)
                action_n = add_noise(action_n)

                self.buffer.add(curr_obs_n, next_obs_n, action_n, reward_n, done_n)

                step += 1
                done = all(done_n)
                terminal = (step >= self.args.max_episode_len)

                for i in self.n:
                    self.agent_update(i)
                
                if done or terminal:
                    break

            if episode % self.args.target_update_rate:
                for i in self.n:
                    self.agent_target_update(i, self.args.tau)
    
    
                
                



