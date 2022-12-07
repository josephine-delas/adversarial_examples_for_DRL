

from agents.ddqn_agent import ddqn_agent

def dqn_train(args, env, device):
    victim = 1 # we are training a victim agent
    agent = ddqn_agent(env, args, device, victim)