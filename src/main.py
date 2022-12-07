import os

import json
import time
import random
import numpy as np
import torch
import gym
import stable_baselines3 as sb3


from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv

from stable_baselines3.common.callbacks import CallbackList

import argparse

if __name__ == "__main__":
  print('ok')
  print('cuad ?' + torch.cuda.is_available())