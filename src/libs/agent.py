from typing import List, Optional, Union

import numpy as np
import torch

from src.libs.communicator import Communicator
from src.libs.controller import Controller
from src.libs.dql import DQN, ExperienceReplayer, ReplayMemory


class Agent:
    # pylint: disable=[R0902(too-many-instance-attributes)]
    """RL agent.

    Attributes:
        controller (src.libs.controller.Controller): Controller object.
        communicator (src.libs.communicator.Communicator): Communicator object.
        policy_net (src.libs.dql.DQN): Policy network
        target_net (Optional[src.libs.dql.DQN]): Target Network. Used for training only. Default to None.
        replayer (Optional[src.libs.dql.ExperienceReplayer]): ExperienceReplayer object. Used for training only.
            Default to None.
        optimiser (Optional[object]): Optimiser object. Used for training only. Default to None.
        memory_size (Optional[int]): Size of the memory tank.
        alpha_initial (Optional[float]): Initial learning rate of the Q-Network.
        alpha_decay (Optional[float]): Decay rate of the Q-Network's learning rate.
        alpha_update (Optional[int]): Number of training steps to update the learning rate.
        epsilon_initial (Optional[float]): Initial value of epsilon.
        epsilon_decay (Optional[float]): Decay rate of epsilon
        epsilon_min (Optional[float]: Minimum value of epsilon.
        _epsilon (float): Current value of epsilon
        _previous_state (List[Union[float, int]]): Previous observed state.
        total_tstep (int): Total number of steps that the agent has been played.
        action (int): Action that the agent takes.
    """

    # pylint: disable=[R0913(too-many-arguments)]
    def __init__(
        self,
        controller: Controller,
        communicator: Communicator,
        policy_net: DQN,
        target_net: Optional[DQN] = None,
        replayer: Optional[ExperienceReplayer] = None,
        optimiser: Optional[torch.optim.Optimizer] = None,
        memory_size: int = 10000,
        alpha_initial: float = 0.001,
        alpha_decay: float = 0.5,
        alpha_update: int = 3000,
        epsilon_initial: float = 1,
        epsilon_decay: float = 0.01,
        epsilon_min: float = 0.003,
    ):

        self.controller = controller
        self.policy_net = policy_net
        self.target_net = target_net
        self.communicator = communicator
        self.memory = ReplayMemory(memory_size)
        self.replayer = replayer
        self.optimiser = optimiser
        self.alpha_initial = alpha_initial
        self.alpha_decay = alpha_decay
        self.alpha_update = alpha_update
        self.epsilon_initial = epsilon_initial
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self._epsilon = self.epsilon_initial
        self._previous_state = None
        self.total_tstep = 0
        self.action = None

    def step(self, observed_state: List[Union[float, int]]):
        """Steps towards the next simulation step.

        Args:
            observed_state (List[Union[float, int]]): Observed state.

        """

        # Choose and send action to Unity
        action = self.controller.choose_action(observed_state, self._epsilon)
        self.communicator.send_data([action, 0])

        self._previous_state = observed_state
        self.action = action

        self.total_tstep += 1

    def replay_experience(
        self,
        observed_state: List[Union[float, int]],
        reward: Union[float, int],
        crash: int,
    ):
        """Replays the stored experience in the memory to learn and update the Q-Network.

        Args:
            observed_state (List[Union[float, int]]): Observed state.
            reward (Union[float, int]): Reward gained.
            crash (int): 0 if not crash, 1 if crash.
        """

        # Store experience in replay memory
        self.memory.push(
            self.replayer.experience(
                torch.tensor([self._previous_state]),
                torch.tensor([self.action]),
                torch.tensor([observed_state]),
                torch.tensor([reward]),
                torch.tensor([crash]),
            )
        )

        # Perform experience replay & update target network by the schedule
        self.replayer.replay(
            self.memory,
            self.policy_net,
            self.target_net,
            self.optimiser,
            self.total_tstep,
        )

    def update(self, i_episode: int):
        """Updates the learning rate and epsilon value.

        Args:
            i_episode (int): Number of the current episode.
        """

        if self.optimiser:

            # Update learning rate
            alpha = self.alpha_initial * (
                self.alpha_decay ** np.floor(self.total_tstep / self.alpha_update)
            )
            for param_group in self.optimiser.param_groups:
                param_group["lr"] = alpha

            # Update epsilon
            self._epsilon = max(
                self.epsilon_min,
                self.epsilon_initial * ((1 - self.epsilon_decay) ** i_episode),
            )
