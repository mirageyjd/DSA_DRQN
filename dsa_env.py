import numpy as np


class DsaCliqueEnv(object):
    def __init__(self, num_user, num_channel, r_fail, r_succeed, r_idle):
        self.num_user = num_user
        self.num_channel = num_channel

        # reward
        self.r_fail = r_fail
        self.r_succeed = r_succeed
        self.r_idle = r_idle

        # timestamp
        self.t = 0

    def reset(self):
        self.t = 0
        obs = np.zeros((self.num_user, self.num_channel), dtype=int)
        return obs

    def step(self, action):
        self.t += 1
        in_use = np.zeros(self.num_channel, dtype=int)
        r = np.zeros(self.num_user)
        obs = np.zeros((self.num_user, self.num_channel), dtype=int)

        for i in range(self.num_user):
            if 0 < action[i] <= self.num_channel:
                in_use[action[i] - 1] += 1
        for i in range(self.num_user):
            if 0 < action[i] <= self.num_channel:
                if in_use[action[i] - 1] > 1:   # conflict
                    r[i] = self.r_fail
                else:                           # succeed
                    r[i] = self.r_succeed
                    obs[i, action[i] - 1] = 1
            else:
                r[i] = self.r_idle

        return obs, r, False, None
