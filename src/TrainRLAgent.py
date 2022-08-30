#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 01:36:46 2019

@author: thepoch
"""
# =============================================================================
# Import required dependencies
# =============================================================================
from __future__ import print_function

import os
import shutil
import socket
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from DQL import DQN, ExperienceReplayer, ReplayMemory, update_learning_rate, experience, calculate_reward
from src.controller import Controller
from OtherFunc import (
    read_spec,
    save_model,
    save_result
)
from src.OtherFunc import plot_progress
from communicator import Communicator

# =============================================================================

"""Choose whether to start a new trainng or continue from the previous one"""
StartNewSim = True

"""Choose the type of controller. 
'RL' => Reinforcement Learning controller,
'HC' => Hand controller (Breaitenberg controller) """
controller_type = "RL"


# =============================================================================
# Learning Parameters
# =============================================================================
gamma = 0.95  # Reward discount factor
epsilon_initial = 1  # Initial value of epsilon
epsilon_min = 0.01  # Minimum possible epsilon (the value it will anneal to)
epsilon_decay = 0.003  # Epsilon decay factor. (Higher -> faster decay rate)
alpha_initial = 0.001  # Initial value of learning rate
alpha_decay = 0.5  # Learning rate decay factor (Hgiher -> slower decay)
batch_size = 64  # The size of the batch of sampled experiences
target_update = 3000  # Interval to update target network
lr_update = 30000  # Interval to step down learning rate
memory_size = 300000  # Capacity of replay memory
layers = [128, 128]  # Number of nodes of 1st and 2nd hidden layers of NN
max_env_steps = 1000  # Maximum training steps of each episode
n_episodes = 10000  # Maximum number of episodes for one training
replay_epoch = 1  # Number of epoch for an experience replay


# =============================================================================
# Other Parameters
# =============================================================================
NumPixelHor = 10  # No. of horizontal pixels to resize the image to
NumPixelVer = 10  # No. of vertical pixels to resize the image to
DeptEstSpeed = 0.1  # Time delay for depth estimation algorithm
n_hidden_states = 3  # No. of hidden states. The array data sent from unity is always
                       # in the following format: [HIDDEN_STATES, OBSERVED_STATES, DISTANCE_READING]
action_space_size = 3  # Size of action space
model_save_interval = 10000  # Interval to save NN model
sma_period_reward = 100  # period to caculate moving average of reward graph
sma_period_loss = 1000  # period to caculate moving average of loss graph
# =============================================================================
# Get Specifications from Unity
# =============================================================================
# mainpath = str(Path(__file__).parents[1])
SpecPath = "" + "../spec.txt"
spec = read_spec(SpecPath, NumPixelHor, NumPixelVer)
InputDim = [spec.values[0, 2], spec.values[0, 1]]
InputType = spec.values[0, 0]


# =============================================================================
# Start New Training or Continue from what was left off
# =============================================================================
# if StartNewSim == True:

"""Initialise parameters, result tanks, and replay memory"""
epsilon = epsilon_initial
alpha = alpha_initial
memory = ReplayMemory(memory_size)
loss_array = []
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
e_c = 0
total_tstep = 0
# image_old = np.ones([NumPixelVer, NumPixelHor])
# Experience = namedtuple(
#     "Experience", ("state", "action", "next_state", "reward", "crash")
# )

"""Create new file path for new experiment"""
ID = datetime.now().strftime("%d%m%Y%H%M")
newpath = "../History"
if not os.path.exists(newpath):
    os.makedirs(newpath)
newpath = newpath + "/" + spec.values[0, 0] + "_" + controller_type + "_" + ID
if os.path.exists(newpath):
    shutil.rmtree(newpath)
os.makedirs(newpath)
modelpath = newpath + "/model"
if not os.path.exists(modelpath):
    os.makedirs(modelpath)

"""Create text file containing learning parameters for this experiment"""
inputs_list = [
    "gamma = " + str(gamma) + "\n",
    "epsilon_initial = " + str(epsilon_initial) + "\n",
    "epsilon_min = " + str(epsilon_min) + "\n",
    "epsilon_decay = " + str(epsilon_decay) + "\n",
    "alpha_initial = " + str(alpha_initial) + "\n",
    "alpha_decay = " + str(alpha_decay) + "\n",
    "target_update = " + str(target_update) + "\n",
    "lr_update = " + str(lr_update) + "\n",
    "memory_size = " + str(memory_size) + "\n",
    "batch_size = " + str(batch_size) + "\n",
    "NN = " + str(layers[0]) + "," + str(layers[1]) + "\n",
    "Outcome:",
]
file = open(newpath + "/info.txt", "w")
file.writelines(inputs_list)
file.close()
spec.to_csv(newpath + "/spec.csv", index=None, sep=",", mode="a")

"""Create connection with Unity"""
host, port = "127.0.0.1", 25001  # Must be identical to the ones in Unity code
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, port))

communicator = Communicator(sock, InputType, InputDim, DeptEstSpeed)
initial_data = communicator.ReceiveData()
communicator.SendData([0, 1])  # Reset the environment
n_observed_state = len(initial_data[0]) - n_hidden_states

"""Initialise neural network model"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(n_observed_state, layers, action_space_size).to(device)
target_net = DQN(n_observed_state, layers, action_space_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(
    params=policy_net.parameters(),
    lr=alpha,
)

"""initialise classes"""
ExperienceReplay = ExperienceReplayer(
    batch_size, replay_epoch, gamma, target_update
)
Agent = Controller(action_space_size, input_dim=InputDim, controller_type=controller_type, policy_net=policy_net)

# elif StartNewSim == False:
#     """Get values and models from what was left off"""
#     e_c = len(result)
#     checkpoint = torch.load(modelpath + "/model_recent.h5")
#     policy_net = DQN(InputDim, layers, truestate, action_space_size).to(device)
#     target_net = DQN(InputDim, layers, truestate, action_space_size).to(device)
#     policy_net.load_state_dict(checkpoint["state_dict"])
#     optimizer = optim.Adam(params=policy_net.parameters(), lr=alpha)
#     optimizer.load_state_dict(checkpoint["optimizer"])
#     target_net.load_state_dict(policy_net.state_dict())





# =============================================================================
# Begin the Training
# =============================================================================
for e in range(e_c, n_episodes):

    """Initialise episodic results"""
    reward_cumu = 0
    tstep = 0
    act = [0, 0, 0]
    positions_tmt = []

    """Initialise the simulation"""
    # communicator.SendData([0, 1])  # Reset the environment

    """Get first set of state"""
    state, rem = communicator.ReceiveData()
    crash = state[0]

    while not crash and tstep < max_env_steps:
        """Choose and send action to Unity"""
        action = Agent.choose_action(state[-n_observed_state:], epsilon)
        communicator.SendData([action, 0])

        """Get next state and other information from Unity"""
        next_state, rem = communicator.ReceiveData()
        crash = next_state[0]
        reward = calculate_reward(next_state)

        """Store experience in replay memory"""
        memory.push(
            experience(
                torch.tensor([state[-n_observed_state:]]),
                torch.tensor([action]),
                torch.tensor([next_state[-n_observed_state:]]),
                torch.tensor([reward]),
                torch.tensor([crash]),
            ),
            rem,
        )

        """Perform experience replay & update target network by the schedule"""
        ExperienceReplay.replay(
            memory, policy_net, target_net, optimizer, total_tstep
        )

        """Trainsition to the next state"""
        state = next_state

        """Append or cumulate results"""
        reward_cumu += reward
        tstep += 1
        total_tstep += 1
        act[action] += 1
        positions_tmt.append(state[2:4])

    """Reset the environment"""
    communicator.SendData([0, 1])

    """Store episodic results in result tank"""
    positions += positions_tmt

    result = pd.concat(
        [
            result,
            pd.DataFrame.from_records(
                [
                    {
                        "Cumulative Reward": reward_cumu,
                        "Time Step": tstep,
                        "Crash": int(crash),
                        "Straight": act[0],
                        "Left Turn": act[1],
                        "Right Turn": act[2],
                    }
                ]
            ),
        ]
    )

    """Save results and model"""
    save_result(positions, ExperienceReplay._loss_record, result, newpath)
    save_model(policy_net, optimizer, modelpath, total_tstep, model_save_interval)

    """Plot progress"""
    plot_progress(
        result,
        ExperienceReplay._loss_record,
        crash,
        e,
        reward_cumu,
        newpath,
        [sma_period_reward, sma_period_loss],
    )

    """Update learning rate"""
    update_learning_rate(alpha_initial, alpha_decay, total_tstep, lr_update, optimizer)

    """Update epsilon"""
    epsilon = max(epsilon_min, epsilon_initial * ((1 - epsilon_decay) ** e))


sock.close()
