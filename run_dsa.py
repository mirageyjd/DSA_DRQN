from dsa_env import DsaCliqueEnv
from agent import Agent
from train_agent import train_agent

device = 'cpu'
num_user = 10
num_channel = 10

env = DsaCliqueEnv(num_user=num_user, num_channel=num_channel, r_fail=-1, r_idle=0, r_succeed=1)
agents = [Agent(env=env, device=device) for i in range(num_user)]
train_agent(env=env, agents=agents, num_it=10000, num_ep=5, max_ts=100, target_update_freq=5, gamma=1.0,
            lstm_hidden_size=128, eps_start=1.0, eps_end=0.01, eps_end_it=1000)
