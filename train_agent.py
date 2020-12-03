import torch
import numpy as np
from tqdm import tqdm
from agent import Agent
import os


def train_agent(env, device, exp_name, agents, num_it, num_ep, max_ts, target_update_freq, gamma, lstm_hidden_size,
                eps_start, eps_end, eps_end_it, beta_start, beta_end):
    """
    env:                environment
    device:             cpu or gpu
    agents:             list of agents, specified by env.num_user

    num_it:             num of iterations
    num_ep:             num of episode in each iteration
    max_ts:             num of time slots in each episode

    target_update_freq: number of iterations token to update target network
    gamma:              dicount factor
    lstm_hidden_size:   the size of hidden state in LSTM layer

    eps_start:          epsilon at the beginning of training
    eps_end:            epsilon at the end of training
    eps_end_it:         number of iterations token to reach ep_end in linearly-annealed epsilon-greedy policy
    """

    log_dir = './' + exp_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = open(log_dir + 'log.txt', 'w+')

    batch_size = num_ep * max_ts
    s_batch = torch.empty((env.num_user, batch_size, env.n_observation), dtype=torch.float).to(device=device)
    s2_batch = torch.empty((env.num_user, batch_size, env.n_observation), dtype=torch.float).to(device=device)
    a_batch = torch.empty((env.num_user, batch_size), dtype=torch.int64).to(device=device)
    r_batch = torch.empty((env.num_user, batch_size), dtype=torch.float).to(device=device)
    h0_batch = torch.empty((env.num_user, batch_size, lstm_hidden_size), dtype=torch.float).to(device=device)
    h1_batch = torch.empty((env.num_user, batch_size, lstm_hidden_size), dtype=torch.float).to(device=device)
    h20_batch = torch.empty((env.num_user, batch_size, lstm_hidden_size), dtype=torch.float).to(device=device)
    h21_batch = torch.empty((env.num_user, batch_size, lstm_hidden_size), dtype=torch.float).to(device=device)
    a = np.zeros(env.num_user, dtype=int)

    for it in tqdm(range(1, num_it + 1)):
        epsilon = eps_start - (eps_start - eps_end * (it - 1) / (eps_end_it - 1)) if it <= eps_end_it else eps_end
        beta = beta_start + (it - 1) * (beta_end - beta_start) / (num_it - 1)

        # sampling from environment
        cnt = 0
        avg_r = 0
        avg_utils = 0
        for ep in range(num_ep):
            s = env.reset()
            h0 = torch.normal(mean=0, std=0.01, size=(env.num_user, lstm_hidden_size)).to(device=device)
            h1 = torch.normal(mean=0, std=0.01, size=(env.num_user, lstm_hidden_size)).to(device=device)
            h20 = h0.clone()
            h21 = h1.clone()
            for t in range(0, max_ts):
                for j in range(env.num_user):
                    a[j], (h20[j], h21[j]) = agents[j].action(s[j], h0[j], h1[j], epsilon, beta)
                s2, r, done, channel_status = env.step(a)

                # collect training samples in batch
                s_batch[:, cnt, :] = torch.from_numpy(s)
                s2_batch[:, cnt, :] = torch.from_numpy(s2)
                a_batch[:, cnt] = torch.from_numpy(a)
                r_batch[:, cnt] = torch.from_numpy(r)
                h0_batch[:, cnt, :] = h0
                h1_batch[:, cnt, :] = h1
                h20_batch[:, cnt, :] = h20
                h21_batch[:, cnt, :] = h21

                avg_r += r.sum() / r.size
                avg_utils += np.sum(channel_status == 1) / channel_status.size
                cnt += 1
                s = s2
                h0 = h20.clone()
                h1 = h21.clone()

        # training
        for j in range(env.num_user):
            agents[j].train(s_batch[j], h0_batch[j], h1_batch[j], a_batch[j], r_batch[j], s2_batch[j], h20_batch[j],
                            h21_batch[j], gamma)

        if it % target_update_freq == 0:
            for j in range(env.num_user):
                agents[j].update_target()

        # print reward
        if it % 100 == 0:
            log_file.write('Iteration {}: avg reward is {:.4f}, channel utilization is {:.4f}\n'.format(it, avg_r / cnt,
                                                                                                      avg_utils / cnt))
    log_file.close()
    for i in range(env.num_user):
        model = agents[i].get_model()
        torch.save(model.state_dict(), log_dir + 'agent_' + str(i) + '.model')


def eval_agent(env, device, exp_name, model_files, num_ep, max_ts, eps_end, lstm_hidden_size):

    log_dir = './' + exp_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = open(log_dir + 'eval_log.txt', 'w+')

    # Load trained networks and build agents
    agents = []
    for i in range(len(model_files)):
        a = Agent(env=env, device=device, lstm_hidden_size=lstm_hidden_size)
        a.load_model_from_state_dict(torch.load(model_files[i]))
        agents.append(a)
    
    epsilon = eps_end

    # Evaluation
    for ep in range(num_ep):
        s = env.reset()
        h0 = torch.normal(mean=0,
                        std=0.01,
                        size=(env.num_user, lstm_hidden_size)).to(device=device)
        h1 = torch.normal(mean=0,
                        std=0.01,
                        size=(env.num_user, lstm_hidden_size)).to(device=device)
        h20 = h0.clone()
        h21 = h1.clone()
        a = np.zeros(env.num_user, dtype=int)
        user_r = np.zeros(env.num_user, dtype=float)
        avg_r = 0.0
        avg_utils = 0.0

        for t in range(max_ts):
            for j in range(env.num_user):
                a[j], (h20[j], h21[j]) = agents[j].action(s[j], h0[j], h1[j], epsilon)
            s2, r, done, channel_status = env.step(a)
            user_r += r

            avg_r += r.sum() / r.size
            avg_utils += np.sum(channel_status == 1) / channel_status.size
            s = s2
            h0 = h20.clone()
            h1 = h21.clone()

        # Calculate the avg reward / timestamp
        user_r /= max_ts
        avg_r /= max_ts
        avg_utils /= max_ts
        log_file.write('Episode {}: Eval results:\n'.format(str(ep)))
        log_file.write('Avg reward: {:.4f} \nAvg channel util: {:.4f}\n'.format(
            avg_r, avg_utils))
        for i in range(len(user_r)):
            log_file.write('User {} avg reward / timestamp: {:.4f}\n'.format(i, user_r[i]))

    log_file.close()
