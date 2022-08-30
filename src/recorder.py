from __future__ import print_function

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch

from OtherFunc import save_model, save_result
from src.OtherFunc import get_moving_average


class Recorder:

    def __init__(self, exp_path: str):

        self.exp_path = exp_path

        self.model_path = self.exp_path + "/model"
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # Initialise episodic record
        self.reward_cumu = 0
        self.tstep = 0
        self.action_count = [0, 0, 0]

        # Initialise global record
        self.position = []
        self.crash = False

        # Initialise result dataframe
        self.result = pd.DataFrame(
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
        self.reward_cumu = 0
        self.tstep = 0
        self.action_count = [0, 0, 0]

    def update(self, reward, action, position, crash):

        self.reward_cumu += reward
        self.tstep += 1
        self.action_count[action] += 1

        self.position.append(position)
        self.crash = bool(crash)

    def save_record(self, loss_record):

        """Store episodic results in result tank"""
        self.result = pd.concat(
            [
                self.result,
                pd.DataFrame.from_records(
                    [
                        {
                            "Cumulative Reward": self.reward_cumu,
                            "Time Step": self.tstep,
                            "Crash": int(self.crash),
                            "Straight": self.action_count[0],
                            "Left Turn": self.action_count[1],
                            "Right Turn": self.action_count[2],
                        }
                    ]
                ),
            ]
        )

        """Save result, loss, and flight path to files"""
        positions_df = pd.DataFrame.from_records(self.position, columns=["x", "z"])
        loss_df = pd.DataFrame(loss_record, columns=["loss"])

        self.result.to_csv(self.exp_path + "/result.csv", index=False, header=True)
        positions_df.to_csv(self.exp_path + "/positions.csv", index=False, header=True)
        loss_df.to_csv(self.exp_path + "/loss.csv", index=False, header=True)

    def save_model(self, policy_net, optimizer, total_step, save_interval):

        """Save the most recet neural network model"""
        policy_net_state = {
            "state_dict": policy_net.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(policy_net_state, self.model_path + "/model_recent.h5")

        """Save neural network model at a specific interval of training steps"""
        s = np.floor(total_step / save_interval)
        torch.save(policy_net.state_dict(), self.model_path + "/model_" + str(int(s + 1)) + ".h5")

    def plot(self, i_episode, loss_record, sma_period_reward, sma_period_loss):

        """Get simple moving average of reward and loss"""
        reward_sma = get_moving_average(sma_period_reward, self.result["Cumulative Reward"].tolist())
        loss_sma = get_moving_average(sma_period_loss, loss_record)

        """assign different marker color depending on who the episode ends"""
        if self.crash:
            marker = "ro"
        else:
            marker = "go"

        """Plot reward and loss graphs"""
        if len(loss_record) > 0:
            plt.subplot(2, 1, 1)
            plt.plot(self.result["Cumulative Reward"], linewidth=0.3)
            plt.plot(reward_sma)
            plt.plot(i_episode, self.reward_cumu, marker)
            plt.subplot(2, 1, 2)
            plt.plot(loss_record, linewidth=0.1)
            plt.plot(loss_sma)
            plt.ylim(0, max(np.array(loss_sma).flatten()) * 1.1)
            plt.savefig(self.exp_path + "/result.png")
            plt.pause(0.0001)
            plt.close()
