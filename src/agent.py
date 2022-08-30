from typing import List, Union

import torch
import numpy as np

from controller import Controller
from communicator import Communicator
from DQL import DQN, ReplayMemory, ExperienceReplayer, experience


class Agent:
    """

    """

    def __init__(self,
                 controller: Controller,
                 policy_net: DQN,
                 target_net: DQN,
                 communicator: Communicator,
                 replayer: ExperienceReplayer,
                 optimiser: torch.optim,
                 memory_size: int,
                 alpha_initial: float,
                 alpha_decay: float,
                 lr_update: int,
                 epsilon_initial: float,
                 epsilon_decay: float,
                 epsilon_min: float
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
        self.lr_update = lr_update
        self.epsilon_initial = epsilon_initial
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self._epsilon = self.epsilon_initial

        self._previous_state = None
        self._total_tstep = 0
        self._action = None

    def step(self, observed_state: List[Union[float, int]]):

        # Choose and send action to Unity
        action = self.controller.choose_action(state=observed_state, epsilon=self._epsilon)
        self.communicator.SendData([action, 0])

        self._previous_state = observed_state
        self._action = action

        self._total_tstep += 1

    def replay_experience(self, observed_state, reward, crash, rem):

        """Store experience in replay memory"""
        self.memory.push(
            experience(
                torch.tensor([self._previous_state]),
                torch.tensor([self._action]),
                torch.tensor([observed_state]),
                torch.tensor([reward]),
                torch.tensor([crash]),
            ),
            rem,
        )

        """Perform experience replay & update target network by the schedule"""
        self.replayer.replay(
            self.memory,
            self.policy_net,
            self.target_net,
            self.optimiser,
            self._total_tstep
        )

    def update(self, i_episode: int):

        # Update learning rate
        alpha = self.alpha_initial * (self.alpha_decay ** np.floor(self._total_tstep / self.lr_update))
        for param_group in self.optimiser.param_groups:
            param_group["lr"] = alpha

        # Update epsilon
        self._epsilon = max(self.epsilon_min, self.epsilon_initial * ((1 - self.epsilon_decay) ** i_episode))
