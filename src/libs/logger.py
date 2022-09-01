import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from src.libs.utils import get_moving_average


class Logger:
    """Logger class to log the simulation results.

    Attributes:
        exp_path (str): Path of the experiment folder to save the results to.
        _model_path (str): Path to model folder
        _reward_cumu (Union[int, float]): Cumulative reward.



    """

    def __init__(self, exp_path: str):

        self.exp_path = exp_path

        self._model_path = self.exp_path + "/models"
        if not os.path.exists(self._model_path):
            os.makedirs(self._model_path)

        # Initialise episodic record
        self._reward_cumu = 0
        self.tstep = 0
        self._action_count = [0, 0, 0]

        # Initialise global record
        self._position: List[List[float]] = []
        self._crash = 0

        # Initialise result dataframe
        self._result = pd.DataFrame(
            columns=[
                "Cumulative Reward",
                "Time Step",
                "Crash",
                "Straight",
                "Left Turn",
                "Right Turn",
            ]
        )

    def reset(self):
        """Resets the result. This is call at the beginning of each episode."""

        self._reward_cumu = 0
        self.tstep = 0
        self._action_count = [0, 0, 0]

    def update(self, reward, action, position, crash):
        """Updates the result."""

        self._reward_cumu += reward
        self.tstep += 1
        self._action_count[action] += 1

        self._position.append(position)
        self._crash = crash

    def save_log(self, loss_record: Optional[list] = None):
        """Save the logged results"""

        # Store episodic results in result tank.
        self._result = pd.concat(
            [
                self._result,
                pd.DataFrame.from_records(
                    [
                        {
                            "Cumulative Reward": self._reward_cumu,
                            "Time Step": self.tstep,
                            "Crash": self._crash,
                            "Straight": self._action_count[0],
                            "Left Turn": self._action_count[1],
                            "Right Turn": self._action_count[2],
                        }
                    ]
                ),
            ]
        )

        # Save result, loss, and flight path to files.
        self._result.to_csv(self.exp_path + "/result.csv", index=False, header=True)
        positions_df = pd.DataFrame.from_records(self._position, columns=["x", "z"])
        positions_df.to_csv(self.exp_path + "/positions.csv", index=False, header=True)

        if loss_record:
            loss_df = pd.DataFrame(loss_record, columns=["loss"])
            loss_df.to_csv(self.exp_path + "/loss.csv", index=False, header=True)

    def save_model(self, policy_net, optimizer, total_step, save_interval):
        """Saves models."""

        # Save the most recent neural network model.
        policy_net_state = {
            "state_dict": policy_net.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(policy_net_state, self._model_path + "/model_recent.h5")

        # Save neural network model at a specific interval of training steps.
        torch.save(
            policy_net.state_dict(),
            self._model_path
            + "/model_"
            + str(int(np.ceil(total_step / save_interval)))
            + ".h5",
        )

    def plot(self, i_episode, loss_record, sma_period_reward, sma_period_loss):
        """Plot graphs of reward and loss."""

        # Get simple moving average of reward and loss.
        reward_sma = get_moving_average(
            sma_period_reward, self._result["Cumulative Reward"].tolist()
        )
        loss_sma = get_moving_average(sma_period_loss, loss_record)

        # assign different marker color depending on who the episode ends.
        if self._crash:
            marker = "ro"
        else:
            marker = "go"

        # Plot reward and loss graphs.
        if len(loss_record) > 0:
            plt.subplot(2, 1, 1)
            plt.plot(self._result["Cumulative Reward"].values, linewidth=0.3)
            plt.plot(reward_sma)
            # plt.plot(i_episode, self._reward_cumu, marker)
            plt.subplot(2, 1, 2)
            plt.plot(loss_record, linewidth=0.1)
            plt.plot(loss_sma)
            plt.ylim(0, None)
            plt.savefig(self.exp_path + "/result.png")
            plt.pause(0.0001)
            plt.close()
