from __future__ import print_function

import random

import numpy as np
import torch


class Agent:
    def __init__(self, action_space_size, input_dim, controller: str):
        self.action_space_size = action_space_size
        self.input_dim = input_dim
        self.controller = controller

    def get_reward(self, data):
        """Calculate reward at current time step"""
        distdiff = data[3]
        crash = bool(data[0])
        reward = np.sign(distdiff) * (1 - crash) - crash
        return reward

    def choose_action(self, state, epsilon, policy_net):

        if self.controller == 'RL':
            controller_func = self._rl_controller

        else:
            controller_func = self._braitenberg_controller

        return int(controller_func(state, epsilon, policy_net))

    def _rl_controller(self, observed_state, epsilon, policy_net):
        """e-greedy strategy"""

        if np.random.random() <= epsilon:
            action = random.randrange(self.action_space_size)
        else:
            with torch.no_grad():
                predict = policy_net(torch.tensor([observed_state]).float())
                action = predict.argmax(dim=1).item()
        return int(action)

    def _braitenberg_controller(self, observed_state, **kwargs):
        """Braitenberg controller"""

        distance_array = observed_state[-np.prod(self.input_dim):]
        middle_index = int((len(distance_array) / self.input_dim[0]) / 2)

        distance_array = np.array(distance_array)
        distance_array = distance_array.reshape([self.input_dim[0], self.input_dim[1]])
        distance_array = distance_array.tolist()

        if (len(distance_array) % 2) != 0:
            for row in distance_array:
                del row[middle_index]

        distance_array = np.array(distance_array)
        distance_array = np.mean(distance_array, 0)

        N = [0, 0]
        for i in range(len(distance_array)):
            N_index = i >= middle_index
            N[N_index] += 1 / (distance_array[i] ** 2)
        action = (N.index(min(N)) + 1) * np.sign(1 - np.average(distance_array))

        return int(action)
