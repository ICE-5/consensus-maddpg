import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


from .agent import Agent
from .replay_buffer import ReplayBuffer
from utils.helper import one_hot, add_noise
from copy import deepcopy

class MADDPG:
    def __init__(self, args, env):    
        self.n = args.num_agents         # num of agents
        self.ad = args.action_dim       # action dim
        self.od = args.obs_dim          # obs dim
        self.batch_size = args.batch_size
        
        print(f'od: {self.od}')
        print(f'ad: {self.ad}')
        self.gamma = args.gamma
        self.env = env
        self.agents = [Agent(args) for i in range(self.n)]
        self.buffer = ReplayBuffer(args)
        self.args = args
    
    def memory_burnin(self):
        # TODO: add burn-in process to buffer
        counter = 0
        while counter < self.args.buffer_burnin:
            done = False
            curr_obs_n = self.env.reset()
            while not done:
                action_n = self.env.action_space.sample()
                next_obs_n, reward_n, done_n, _ = self.env.step(action_n)

                action_n = one_hot(action_n, self.args.action_dim)
                self.buffer.add(curr_obs_n, next_obs_n, action_n, reward_n, done_n)

                curr_obs_n = deepcopy(next_obs_n)
                counter +=1



    def agent_critic_loss(self, agent_idx, curr_obs_n, next_obs_n, action_n, reward_n):
        # calculate y
        o = next_obs_n
        a = []
        for i, agent in enumerate(self.agents):
            curr_obs = curr_obs_n[:, i*self.ad : (i+1)*self.ad]
            action = agent.get_action(curr_obs, is_target=True)
            a.append(action)
        y = torch.cat([o, a]).flatten()
        y = self.agents[agent_idx].get_q(y, is_target=True)
        y = self.gamma * y + reward_n[:, agent_idx]

        # calculate goal_y
        goal_o = curr_obs_n
        goal_a = action_n
        goal_y = torch.cat([goal_o, goal_a])
        goal_y = self.agents[agent_idx].get_q(y, is_target=False)
        
        return F.mse_loss(goal_y, y)
    

    def agent_actor_loss(self, agent_idx, curr_obs_n, action_n):
        o = curr_obs_n
        a = action_n

        agent_curr_obs = curr_obs_n[:, agent_idx*self.od : (agent_idx+1)*self.od]
        agent_action = self.agents[agent_idx].get_action(agent_curr_obs, is_target=False)
        a[:, agent_idx*self.ad : (agent_idx+1)*self.ad] = agent_action

        loss = - self.agents[agent_idx].get_q(torch.cat([o, a]), is_target=False).mean()
        return loss

    
    def agent_update(self, agent_idx):
        # get minibatch sample
        curr_obs_n, next_obs_n, action_n, reward_n, _ = self.buffer.sample_minibatch(self.batch_size)
        # update critic
        agent_critic_loss = self.agent_critic_loss(agent_idx, curr_obs_n, next_obs_n, action_n, reward_n)
        self.agents[agent_idx].critic_optim.zero_grad()
        agent_critic_loss.backward()
        self.agents[agent_idx].critic_optim.step()
        # update actor
        agent_actor_loss = self.agent_actor_loss(agent_idx, curr_obs_n, action_n)
        self.agents[agent_idx].actor_optim.zero_grad()
        agent_actor_loss.backward()
        self.agents[agent_idx].actor_optim.step()

        return agent_critic_loss.item(), agent_actor_loss.item()
    

    def agent_target_update(self, agent_idx, tau):
        self.agents[agent_idx].target_update(self.args.tau, actor=True)
        self.agents[agent_idx].target_update(self.args.tau, actor=False)


    def train(self):
        for episode in range(self.args.num_episodes):
            step = 0
            curr_obs_n = self.env.reset()
            while True:
                # action_n = [agent.get_action(curr_obs_n[:, i*self.od : (i+1)*self.od], decode=True) for i, agent in enumerate(self.agents)]
                action_n = [agent.get_action(curr_obs_n[i], decode=True) for i, agent in enumerate(self.agents)]
                next_obs_n, reward_n, done_n, _ = self.env.step(action_n)

                action_n = one_hot(action_n, self.args.action_dim)
                action_n = add_noise(action_n)

                self.buffer.add(curr_obs_n, next_obs_n, action_n, reward_n, done_n)
                curr_obs_n = deepcopy(next_obs_n)

                step += 1
                done = all(done_n)
                terminal = (step >= self.args.max_episode_len)

                critic_loss_list, actor_loss_list = [], []

                for i in self.n:
                    critic_loss, actor_loss = self.agent_update(i)
                    critic_loss_list.append(critic_loss)
                    actor_loss_list.append(actor_loss)
                
                for i in self.n:
                    self.agent_target_update(i, self.args.tau)
                
                if done or terminal:
                    if episode % 100 == 0:
                        print(f'episode: {episode}, step: {step}\t\t critic loss: {np.sum(critic_loss_list)}\tactor loss: {np.sum(actor_loss_list)}')
                    break
    
            
    
                
                



