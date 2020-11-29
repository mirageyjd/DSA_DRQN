import torch
import numpy as np
from tqdm import tqdm


def train_agent(env, agents, num_it, num_ep, max_ts, target_update_freq, gamma, lstm_hidden_size, eps_start, eps_end,
                eps_end_it):
    """
    env:                environment
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

    batch_size = num_ep * max_ts
    s_batch = torch.empty((env.num_user, batch_size, env.n_observation), dtype=torch.int)
    s2_batch = torch.empty((env.num_user, batch_size, env.n_observation), dtype=torch.int)
    a_batch = torch.empty((env.num_user, batch_size), dtype=torch.int64)
    r_batch = torch.empty((env.num_user, batch_size), dtype=torch.float)
    h0_batch = torch.empty((env.num_user, batch_size, lstm_hidden_size), dtype=torch.float)
    h1_batch = torch.empty((env.num_user, batch_size, lstm_hidden_size), dtype=torch.float)
    h20_batch = torch.empty((env.num_user, batch_size, lstm_hidden_size), dtype=torch.float)
    h21_batch = torch.empty((env.num_user, batch_size, lstm_hidden_size), dtype=torch.float)
    a = np.zeros(env.num_user, dtype=int)

    for it in tqdm(range(1, num_it + 1)):
        epsilon = eps_start - (eps_start - eps_end * (it - 1) / (eps_end_it - 1)) if it <= eps_end_it else eps_end

        # sampling from environment
        cnt = 0
        avg_r = 0
        avg_utils = 0
        for ep in range(num_ep):
            s = env.reset()
            h0 = torch.normal(mean=0, std=0.01, size=(env.num_user, lstm_hidden_size))
            h1 = torch.normal(mean=0, std=0.01, size=(env.num_user, lstm_hidden_size))
            h20 = h0.clone()
            h21 = h1.clone()
            for t in range(0, max_ts):
                for j in range(env.num_user):
                    a[j], (h20[j], h21[j]) = agents[j].action(s[j], h0[j], h1[j], epsilon)
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
        if it % 2 == 0:
            print('Iteration {}: avg reward is {:.4f}, channel utilization is {:.4f}'.format(it, avg_r / cnt, avg_utils / cnt))
