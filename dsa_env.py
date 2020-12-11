import numpy as np


class DsaCliqueEnv(object):
    def __init__(self, num_user, num_channel, r_fail, r_succeed, r_idle, r_fairness=None):
        self.num_user = num_user
        self.num_channel = num_channel

        # reward
        self.r_fail = r_fail
        self.r_succeed = r_succeed
        self.r_idle = r_idle

        # fairness-aware reward
        # If a user has occupied a channel for x consecutive time slots
        # reward of success will be penalized by a factor (sigmoid function)
        # When r_fail = 0 and r_succeed = 1, reward of success will drop to 0.5 at x = f_thre
        # x will be cleared if 1. switch to another channel 2. idle 3. conflict
        self.f_thre = None
        if r_fairness is not None and r_fairness > 0:
            self.delta = 1 + np.exp(-1)
            self.f_thre = r_fairness
        self.history = np.zeros((num_user, 2), dtype=int)

        # space
        self.n_action = num_channel + 1
        self.n_observation = num_channel + 2

        # timestamp
        self.t = 0

    def r_succeed_fair(self, x):
        if self.f_thre is not None:
            tmp = (1 / (1 + np.exp(x / self.f_thre - 1)) - 0.5) * self.delta + 0.5
            return (self.r_succeed - self.r_fail) * tmp + self.r_fail
        else:
            return self.r_succeed

    def reset(self):
        self.t = 0
        self.history = np.zeros((self.num_user, 2), dtype=int)
        obs = np.zeros((self.num_user, self.n_observation), dtype=float)
        obs[:, 0] = 1
        return obs

    def step(self, action):
        self.t += 1
        in_use = np.zeros(self.num_channel, dtype=int)
        r = np.zeros(self.num_user)
        obs = np.zeros((self.num_user, self.n_observation), dtype=float)

        for i in range(self.num_user):
            obs[i, action[i]] = 1
            if action[i] > 0:
                in_use[action[i] - 1] += 1
        for i in range(self.num_user):
            if action[i] > 0:
                if in_use[action[i] - 1] > 1:   # conflict
                    r[i] = self.r_fail
                    self.history[i, 0] = 0
                else:                           # succeed
                    if self.history[i, 0] == action[i]:
                        r[i] = self.r_succeed_fair(self.history[i, 1])
                        self.history[i, 1] += 1
                    else:
                        r[i] = self.r_succeed
                        self.history[i, 0] = action[i]
                        self.history[i, 1] = 1
                    obs[i, -1] = 1
            else:
                r[i] = self.r_idle
                self.history[i, 0] = 0

        return obs, r, False, in_use
