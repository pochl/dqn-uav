import random
from collections import namedtuple
from typing import List, Tuple, Union, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def calculate_reward(state: List[Union[float, int]]) -> Union[int, float]:
    """Calculates reward at current time step

    Args:
        state (List[Union[float, int]]): State.

    Return:
        Union[int, float]: Reward gained.
    """

    dist_diff = state[3]
    crash = bool(state[0])
    reward = np.sign(dist_diff) * (1 - crash) - crash
    return reward


class DQN(nn.Module):
    """Deep-Q Network Model.

    Attributes:
        n_observed_state (int): Number of observed state.
        layers (Tuple[int, int]): (NUMBER OF NODES AT 1ST LAYER, NUMBER OF NODES AT 2ND LAYER)
        action_space_size (int): The number of the possible actions for the agent to take.

    """

    def __init__(
        self, n_observed_state: int, layers: Tuple[int, int], action_space_size: int
    ):
        """Initialises the model."""

        super().__init__()
        self.fc1 = nn.Linear(in_features=n_observed_state, out_features=layers[0])
        self.fc2 = nn.Linear(in_features=layers[0], out_features=layers[1])
        self.out = nn.Linear(in_features=layers[1], out_features=action_space_size)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Feeds forward.

        Args:
            tensor (torch.Tensor): The tensor containing the values.

        Returns:
            torch.Tensor: The tensor containing the values.
        """

        tensor = F.relu(self.fc1(tensor))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.out(tensor)

        return tensor


class ReplayMemory:
    """Replay memory class. This class is used to manage the memory that will be replayed to
    train the model.

    Attributes:
        capacity (int): Capacity of the memory.
        _memory (List[namedtuple]): List of the experience in the memory.
        _push_count (int): The number of times that the memory has been push.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self._memory = []
        self._push_count = 0

    def push(self, experience: NamedTuple):
        """Pushes the experience into the memory.

        Args:
            experience (namedtuple): The experience tuple.
        """

        if len(self._memory) < self.capacity:
            # add new experience.
            self._memory.append(experience)
        else:
            # push the oldest one out if memory is full.
            self._memory[self._push_count % self.capacity] = experience
        self._push_count += 1

    def sample(self, batch_size: int) -> List[NamedTuple]:
        """Samples a batch of experience.

        Args:
            batch_size (int): The size of the batch to be sampled.

        Returns:
            List[namedtuple]: List of the sampled experiences.
        """

        return random.sample(self._memory, batch_size)

    def can_provide_sample(self, batch_size: int) -> bool:
        """Checks if there's enough experiences in memory to be sampled.

        Args:
            batch_size (int): The size of the batch to be sampled.

        Returns:
            bool: Whether there's enough experiences in memory to be sampled.
        """

        return len(self._memory) >= batch_size


class ExperienceReplayer:
    """Experience replayer class. This class is used to perform the experience replay to train the model.

    Attributes:
        batch_size (int): Number of the experiences that will be used to train the model at each epoch.
        replay_epoch (int): Number of epoch to train the model for.
        gamma (float): Discounted reward factor.
        target_update (int): At every X simulation steps, the target net work will be updated.
        loss_record (List[float]): The list of loss values.
        experience (namedtuple): The experience tuple.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
        self, batch_size: int, replay_epoch: int, gamma: float, target_update: int
    ):
        self.batch_size = batch_size
        self.replay_epoch = replay_epoch
        self.gamma = gamma
        self.target_update = target_update

        self.loss_record: List[float] = []

        self.experience = namedtuple(
            "experience", ("state", "action", "next_state", "reward", "crash")
        )

    def extract_tensors(
        self, experiences: List[NamedTuple]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts the experiences into tensors.

        Args:
            experiences (List[namedtuple]): The list of experiences.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,torch.Tensor]): Tuple of tensors,
                (state, action, reward, next_state, crash)
        """

        batch = self.experience(*zip(*experiences))
        ten_1 = torch.cat(batch.state)
        ten_2 = torch.cat(batch.action)
        ten_3 = torch.cat(batch.reward)
        ten_4 = torch.cat(batch.next_state)
        ten_5 = torch.cat(batch.crash)

        return ten_1, ten_2, ten_3, ten_4, ten_5

    @staticmethod
    def q_current(
        policy_net: DQN, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Gets the current q value. I.E. Q(s_t,a_t).

        Args:
            policy_net (DQN): Policy network.
            states (torch.Tensor): Tensors of states.
            actions (torch.Tensor): Tensors of actions.

        Returns:
            torch.Tensor: Tensor of Q values.
        """

        return policy_net(states.float()).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def q_next(
        target_net: DQN, next_states: torch.Tensor, crashes: torch.Tensor
    ) -> torch.Tensor:
        """Gets the next Q value. I.E.  max(Q(s_t+1,a_t+1)).

        Args:
            target_net (DQN): Policy network.
            next_states (torch.Tensor): Tensors of next states.
            crashes (torch.Tensor): Tensors of crashes.

        Returns:
            torch.Tensor: Tensor of Q values.
        """

        # final_state_locations = crashes.eq(1).type(torch.bool)
        # non_final_state_locations = final_state_locations == False
        non_final_state_locations = crashes.eq(0).type(torch.bool)
        non_final_state = next_states[non_final_state_locations].float()
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(ExperienceReplayer.device)
        values[non_final_state_locations] = (
            target_net(non_final_state).max(dim=1)[0].detach()
        )

        return values

    def replay(
        self,
        memory: ReplayMemory,
        policy_net: DQN,
        target_net: DQN,
        optimiser: torch.optim.Optimizer,
        total_tstep: int,
    ):
        """Replays the experiences.

        Args:
            memory (ReplayMemory): Replay memory class.
            policy_net (DQN): Policy network.
            target_net (DQN): Target network.
            optimiser (torch.optim.Optimizer): Optimiser.
            total_tstep (int): Number of total simulation step that has been run.
        """

        if memory.can_provide_sample(self.batch_size):
            # Update target network if it's time for update.
            if not total_tstep % self.target_update:
                target_net.load_state_dict(policy_net.state_dict())
                print("target network updated")

            # Perform experience replay.
            for _ in range(self.replay_epoch):
                experiences = memory.sample(self.batch_size)  # Sample experiences
                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    crashes,
                ) = ExperienceReplayer.extract_tensors(self, experiences)
                current_q_values = ExperienceReplayer.q_current(
                    policy_net, states, actions
                )
                next_q_values = ExperienceReplayer.q_next(
                    target_net, next_states, crashes
                )
                target_q_values = (next_q_values * self.gamma) + rewards

                # Get Huber loss and perform gradient descend
                loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                self.loss_record.append(loss.item())
