#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 01:36:46 2019

@author: thepoch
"""
from __future__ import print_function

import glob
import os
import re
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.libs.dql import DQN
from src.libs.controller import Controller
from src.libs.communicator import Communicator


def numericalSort(value):
    """Function for sorting file name by number"""
    numbers = re.compile(r"(\d+)")
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


"""Choose the type of controller. 
'RL' => Reinforcement Learning controller,
'HC' => Hand controller (Breaitenberg controller) """
controller_type = "RL"
# =============================================================================
# Testing Parameters
# =============================================================================
Folder_Name = "**Visual_RL_210420200111"  # File name of the experiment folder
max_env_steps = 220  # Max episode steps
n_episodes = 20  # Max number of episode for each test
DeptEstSpeed = 0.1
epsilon = 0  # Set to 0 for testing
action_space_size = 3
n_observed_states = 3  # No. of observed states excluding LiDAR/image. The array data sent from unity is always
                       # in the following format: [HIDDEN_STATES, OBSERVED_STATES, DISTANCE_READING]


# =============================================================================
# Initialsing
# =============================================================================
"""Get paths"""
mainpath = str(Path(__file__).parents[1])
path = mainpath + "/History/" + Folder_Name
modelpath = path + "/model"

"""Read info and spec of the experiment"""
info = pd.read_csv(path + "/info.txt", sep=" ", header=None)
NN = info.loc[info[0] == "NN"].values[0, 2].split(",")
layers = [int(NN[0]), int(NN[1])]
spec = pd.read_csv(path + "/spec.csv", sep=",")
InputDim = [spec.values[0, 2], spec.values[0, 1]]
InputType = spec.values[0, 0]

"""Initialise parameters, result tanks, and replay memory"""
total_observed_states = n_observed_states + np.prod(InputDim)

epsilon = 0

total_tstep = 0
image_old = np.ones(InputDim)

"""Create new file path in current experiment's folder for test result"""
testpath = path + "/test"
if not os.path.exists(testpath):
    os.makedirs(testpath)

"""Initialise neural network model"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(InputDim, layers, n_observed_states, action_space_size).to(device)

"""Create connection with Unity"""
host, port = "127.0.0.1", 25001  # Must be identical to the ones in Unity code
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, port))

"""initialise classes"""
transfer_data = Communicator(sock, InputType, InputDim, DeptEstSpeed, n_observed_states)
Agent = Controller(action_space_size, input_dim=InputDim, controller_type=controller_type, policy_net=policy_net)

# =============================================================================
# Begin the Testing
# =============================================================================
for filename in sorted(glob.glob(os.path.join(modelpath, "*.h5")), key=numericalSort):
    """Iterate the models in the model folder to test one by one"""
    file = filename.split("/")[-1]
    file = file[:-3]
    print(file)
    policy_net = DQN(InputDim, layers, n_observed_states, action_space_size).to(device)
    policy_net.load_state_dict(torch.load(filename))
    policy_net.eval()

    """Initialise result tanks for each model"""
    positions = []
    result = pd.DataFrame(
        columns=[
            "Cumulative Reward",
            "Time Step",
            "Crash",
            "Straight",
            "Left Turn",
            "Right Turn",
        ]
    )
    for e in range(0, n_episodes):
        print(e)
        """Initialise episodic results"""
        reward_cumu = 0
        tstep = 0
        act = [0, 0, 0]
        positions_tmt = []

        """Initialise the simulation"""
        transfer_data.receive_data()
        transfer_data.send_data([0, 1])  # Reset the environment

        """Get first set of state"""
        state, _ = transfer_data.receive_data()
        crash = state[0]

        while not crash and tstep < max_env_steps:
            """Choose and send action to Unity"""
            action = Agent.choose_action(state[-total_observed_states:], epsilon)
            transfer_data.send_data([action, 0])

            """Get next state and other information from Unity"""
            next_state, rem = transfer_data.receive_data()
            crash = next_state[0]
            reward = Agent.get_reward(next_state)

            """Trainsition to the next state"""
            state = next_state

            """Append or cumulate results"""
            reward_cumu += reward
            tstep += 1
            act[action] += 1
            total_tstep += 1
            positions_tmt.append(state[2:4])

        """Reset the environment"""
        transfer_data.send_data([0, 1])

        """Store episodic results in result tank"""
        positions += positions_tmt
        result = result.append(
            {
                "Cumulative Reward": reward_cumu,
                "Time Step": tstep,
                "Crash": int(crash),
                "Straight": act[0],
                "Left Turn": act[1],
                "Right Turn": act[2],
            },
            ignore_index=True,
        )

        """Save results"""
        positionsdf = pd.DataFrame.from_records(positions, columns=["x", "z"])
        result.to_csv(
            testpath + "/result_test_" + file + ".csv", index=False, header=True
        )
        positionsdf.to_csv(
            testpath + "/positions_test_" + file + ".csv", index=False, header=True
        )

sock.close()
