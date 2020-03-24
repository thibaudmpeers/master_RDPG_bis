import numpy as np
import random
import torch


class MemoryBuffer:

    def __init__(self, size, number_steps, observation_dim, action_dim):
        self.max_size = size
        self.number_steps = number_steps
        self.obs_dim = observation_dim
        self.action_dim = action_dim
        self.index = self.max_size
        self.len = 0
        self.batch_size = 32

        self.actions_buffer = torch.zeros((self.max_size, self.number_steps, self.action_dim), dtype=torch.float32)
        self.rewards_buffer = torch.zeros((self.max_size, self.number_steps, 1), dtype=torch.float32)
        self.observations_buffer = torch.zeros((self.max_size, self.number_steps, self.obs_dim), dtype=torch.float32)
        self.inputs_buffer = torch.zeros((self.max_size, self.number_steps, self.obs_dim + self.action_dim + 1), dtype=torch.float32)
        self.target_values_buffer = torch.zeros((self.max_size, self.number_steps, 1), dtype=torch.float32)

    def new_epsisode(self):
        if self.len != self.max_size:
            self.len += 1
        if self.index < self.max_size - 1:
            self.index += 1
        else:
            self.index = 0

    def save_trasition(self, observation, action, reward, inputs, step):
        self.observations_buffer[self.index, step, :] = observation.detach()
        self.actions_buffer[self.index, step, :] = action.detach()
        self.rewards_buffer[self.index, step, :] = reward.detach()
        self.inputs_buffer[self.index, step, :] = inputs.detach()

    def save_target_value(self, target_value, step):
        self.target_values_buffer[self.index, step, :] = target_value.detach()

    def sample(self):
        count = min(self.len, self.batch_size)
        batch = random.sample(range(self.len), count)

        s_arr = self.observations_buffer[batch]
        a_arr = self.actions_buffer[batch]
        r_arr = self.rewards_buffer[batch]
        inputs_arr = self.inputs_buffer[batch]
        target_values = self.target_values_buffer[batch]

        return s_arr, a_arr, r_arr, inputs_arr, target_values, count

    def len(self):
        return self.index
