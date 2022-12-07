import torch
import torch.nn as nn
import random
import numpy as np
import math
import sys

from torchviz import make_dot

from utils.replay_memory import ReplayBuffer
from agents.base_agent import base_agent
from agents.models import DQNnet, CnnDQN

class epsilon_schedule:
    '''
    Espilon decay function 
    '''
    def __init__(self, total_timesteps, epsilon_start=1., epsilon_final=0.01):
        self.total_timesteps = total_timesteps
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
    
    def get_value(self, timestep):
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(-6. * timestep / self.total_timesteps)

class ddqn_agent(base_agent):
    def __init__(self, env, args, device, name, victim):
        '''
        args -> victim_agent_mode, attacker_agent_mode, env_name, game_mode (base_agent arguments)
             -> lr, buffer_size, init_ratio, gamma, train_freq, target_network_update_freq, batch_size (ddqn arguments)
        '''
        base_agent.__init__(self, args, name, victim)
        self.env = env
        self.args = args 
        self.device = device
        print('self.env.observation_space.shape')
        print(self.env.observation_space.shape)
        # network
        self.current_model = CnnDQN(self.env.observation_space.shape,self.env.action_space.n) # TODO implement DQNnet

        # target network
        self.target_model = CnnDQN(self.env.observation_space.shape,self.env.action_space.n)

        # device
        self.current_model.to(self.device)
        self.target_model.to(self.device)

        # optimizer
        self.optimizer = torch.optim.Adam(self.current_model.parameters(), lr=self.args.lr)

        # replay memory
        self.replay_buffer = ReplayBuffer(self.args.buffer_size)

        # exploration schedule
        self.explore_eps = self.args.init_ratio
        #self.exploration_schedule = linear_schedule(int(self.args.total_timesteps * self.args.exploration_fraction), \
        #                                            self.args.final_ratio, self.args.init_ratio) TODO implement linear_schedule
        
        self.target_model.load_state_dict(self.current_model.state_dict())

    def act(self, state, epsilon):
        '''
        Choose action using deep q network
        '''
        with torch.no_grad():
            state = torch.FloatTensor(state.copy()).unsqueeze(0)
            action_value = self.current_model.forward(state) # Q value function from DQN
        action = torch.argmax(action_value) if random.random() > epsilon else torch.randint(0, action_value.shape[1], (1,))
        return action.reshape(1,1).item()

    def remember(self, state, action, reward, next_state, done):
        '''
        Store timestep in memory buffer
        '''
        self.replay_buffer.push(state, action, reward, next_state, done)

    def compute_td_loss(self, samples):
        '''
        Compute loss in order to update the network, + gradient backpropagation
        '''
        state, action, reward, next_state, done = self.replay_buffer.sample(self.args.batch_size)

        states = torch.FloatTensor(np.float32(state)).to(self.device) # dim = (batch_size, n_features)
        next_states = torch.FloatTensor(np.float32(next_state)).to(self.device)
        actions = torch.LongTensor(action).to(self.device) # dim = (batch_size)
        rewards = torch.FloatTensor(reward).to(self.device) # dim = (batch_size)
        dones = torch.FloatTensor(done).to(self.device) # dim = (batch_size)

        q_values      = self.current_model(states)
        q_value          = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            next_q_value     = next_q_values.max(1)[0]
            expected_q_value = rewards + self.args.gamma * next_q_value * (1 - dones)
        
        loss = (q_value - expected_q_value.data).pow(2).mean()

        self.optimizer.zero_grad()

        print('optim zero grad')
        make_dot(loss, params=dict(list(self.current_model.named_parameters()))).render("loss_torchivz.png", format="png")
        print(loss)
        sys.stdout.flush()
        loss.backward()
        
        print('loss backward')
        sys.stdout.flush()
        self.optimizer.step()
        
        return loss

    def update_agent(self, timestep):
        if timestep % self.args.train_freq == 0:
            # update model weights at the wanted frequency
            batch_samples = self.replay_buffer.sample(self.args.batch_size)
            td_loss = self.compute_td_loss(batch_samples)
            print('loss computed')
            sys.stdout.flush()
            
        
        if timestep % self.args.target_network_update_freq == 0:
            # update target network regularly
            self.target_model.load_state_dict(self.current_model.state_dict())
        
        return td_loss

    
        
            