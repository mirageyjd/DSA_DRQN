from dsa_env import DsaCliqueEnv
from agent import Agent
from train_agent import train_agent, eval_agent
import argparse

device = 'cpu'
exp_name = 'exp_1'
num_user = 10
num_channel = 6
lstm_hidden_size = 64
model_files = ['./' + exp_name + '/agent_' +
               str(i) + '.model' for i in range(num_user)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training or Evaluation of DSA-DRQN algorithm')
    parser.add_argument('--mode', type=str, default='eval',
                        help="'eval' for evaluation, 'train' for training, default 'eval'")
    args = parser.parse_args()

    if args.mode == 'train':
        env = DsaCliqueEnv(num_user=num_user, num_channel=num_channel, r_fail=-1, r_idle=0, r_succeed=1)
        agents = [Agent(env=env, device=device, lstm_hidden_size=lstm_hidden_size) for i in range(num_user)]
        train_agent(env=env, device=device, exp_name=exp_name, agents=agents, num_it=10000, num_ep=5, max_ts=100,
                    target_update_freq=5, gamma=1.0, lstm_hidden_size=lstm_hidden_size, eps_start=1.0, eps_end=0.01,
                    eps_end_it=1000, beta_start=1, beta_end=20)
    elif args.mode == 'eval':
        env = DsaCliqueEnv(
            num_user=num_user, num_channel=num_channel, r_fail=-1, r_idle=0, r_succeed=1)
        eval_agent(
            env=env, device=device, exp_name=exp_name, model_files=model_files, num_ep=5,
            max_ts=100, eps_end=0.01, lstm_hidden_size=lstm_hidden_size)
    else:
        raise Exception("Unsupported mode.")
