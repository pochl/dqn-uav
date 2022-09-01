import random
from typing import List, Tuple, Union

import numpy as np
import torch

from src.libs.dql import DQN


class Controller:
    """Controller class. It contains different typr of controllers and also the function to choose the action.

    Attributes:
        action_space_size (int): The number of possible actions for the agent to take.
        input_dim (Tuple[int, int]): (vertical_dimension, horizontal_dimension).
        controller_type (str): 'AI' = Use the reinforcement learning algorithm. 'RB' = Use the rule-based algorithm.
        policy_net (DQN): Policy network.
    """

    def __init__(
        self,
        action_space_size: int,
        input_dim: Tuple[int, int],
        controller_type: str,
        policy_net: DQN,
    ):
        self.action_space_size = action_space_size
        self.input_dim = input_dim
        self.controller_type = controller_type
        self.policy_net = policy_net

    def choose_action(
        self, observed_state: List[Union[float, int]], epsilon: float
    ) -> int:
        """Choose the action

        Args:
            observed_state (List[Union[float, int]]): Observed state.
            epsilon (float): Epsilon value for epsilon-greedy strategy.

        Returns:
            int: Action to take.

        Raises:
            ValueError: Invalid controller_type
        ."""

        if self.controller_type == "AI":
            controller_func = self._rl_controller

        elif self.controller_type == "RB":
            controller_func = self._rb_controller

        else:
            raise ValueError("Invalid controller_type")

        return int(controller_func(observed_state, epsilon))

    def _rl_controller(
        self, observed_state: List[Union[float, int]], epsilon: float
    ) -> int:
        """Chooses action using RL algorithm with epsilon-greedy strategy

        Args:
            observed_state (List[Union[float, int]]): Observed state.
            epsilon (float): Epsilon value for epsilon-greedy strategy.

        Returns:
            int: Action to take.
        """

        if np.random.random() <= epsilon:
            action = random.randrange(self.action_space_size)
        else:
            with torch.no_grad():
                predict = self.policy_net(torch.tensor([observed_state]).float())
                action = predict.argmax(dim=1).item()
        return int(action)

    # pylint: disable=[W0613(unused-argument)
    def _rb_controller(self, observed_state: List[Union[float, int]], **kwargs) -> int:
        """Chooses action using rule-based algorithm with Braitenberg algorithm.

        Args:
            observed_state (List[Union[float, int]]): Observed state.
            kwargs: extra arguments.

        Returns:
            int: Action to take.
        """

        distance_list = observed_state[-int(np.prod(self.input_dim)):]
        middle_index = int((len(distance_list) / self.input_dim[0]) / 2)

        distance_array = np.array(distance_list).reshape(
            [self.input_dim[0], self.input_dim[1]]
        )

        if len(distance_array) % 2:
            for row in distance_array:
                del row[middle_index]

        mean_distance_array = np.mean(distance_array, 0)

        rays_count = [0, 0]
        for idx, _ in enumerate(mean_distance_array):
            ray_index = idx >= middle_index
            rays_count[ray_index] += 1 / (mean_distance_array[idx] ** 2)
        action = (rays_count.index(min(rays_count)) + 1) * np.sign(
            1 - np.average(mean_distance_array)
        )

        return int(action)
