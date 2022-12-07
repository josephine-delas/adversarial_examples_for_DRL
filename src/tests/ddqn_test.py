import matplotlib.pyplot as plt
import numpy as np
import math
import sys

import gym
import ale_py
from agents.ddqn_agent import ddqn_agent
from common.wrappers import ImageToPyTorch, AtariWrapper



def plot(frame_idx, rewards, losses, path_save):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.savefig(path_save, format='svg', dpi=1000)



class Args:
  def __init__(self):
    self.victim_agent_mode = 'dqn'
    self.attacker_agent_mode = 'dqn'
    self.env_name = 'atari'
    self.game_mode = 'normal'
    self.lr = 1e-4
    self.buffer_size = 1000
    self.init_ratio = 0.1
    self.gamma = 0.99
    self.train_freq = 1
    self.target_network_update_freq = 1000
    self.batch_size = 32


if __name__ == '__main__':
    print('Start...')
    print(gym.__version__)
    print(gym.envs.registry.keys())
    args = Args()

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 10000

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

    env = gym.make('ALE/Pong-v5')
    env = AtariWrapper(env)
    env = ImageToPyTorch(env)

    num_frames = 100
    batch_size = 32
    gamma      = 0.99

    losses = []
    all_rewards = []
    episode_reward = 0

    agent = ddqn_agent(env, args, 'cpu', 'game', 1)

    state = env.reset()
    print('State shape : ')
    print(state.shape)
    
    print('Beginning training...')
    sys.stdout.flush()
    '''
    for frame_idx in range(1, num_frames + 1):
        print(frame_idx)
        sys.stdout.flush()
        epsilon = epsilon_by_frame(frame_idx)
        action = agent.act(state, epsilon)
        
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        
        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0
            
        if len(agent.replay_buffer) > batch_size:
            loss = agent.update_agent(frame_idx)
            losses.append(loss.data.item())
        
        if frame_idx%1000==0:
            print('Plotting results...')
            plot(num_frames, all_rewards, losses, path_save='/home/jdelas/projects/def-fcuppens/jdelas/figures/test_ddqn_atari.svg')

        
    '''    
    
    print('Done')
