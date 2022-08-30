#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 21:32:08 2020

@author: thepoch
"""
from __future__ import print_function

import re

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import yaml


def load_yaml(path: str):
    with open(path) as file:
        yaml_dict = yaml.load(file, Loader=yaml.FullLoader)
    return yaml_dict


def write_yaml(data, path):

    with open(path, 'w') as file:
        yaml.dump(data, file)


def read_spec(path: str):
    """Read specification from Unity"""

    with open(path) as f:
        lines = f.read().splitlines()

    keys = lines[0].split(" ")
    values = lines[1].split(" ")
    values[1:] = [int(i) for i in values[1:]]

    return dict(zip(keys, values))


def get_moving_average(period, values):
    """Get simple moving average"""
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = (
            values.unfold(dimension=0, size=period, step=1)
            .mean(dim=1)
            .flatten(start_dim=0)
        )
        moving_avg = torch.cat((torch.zeros(period - 1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()


def save_result(positions, loss_array, result, newpath):
    """Save result, loss, and flight path to files"""
    positionsdf = pd.DataFrame.from_records(positions, columns=["x", "z"])
    lossdf = pd.DataFrame(loss_array, columns=["loss"])
    result.to_csv(newpath + "/result.csv", index=False, header=True)
    positionsdf.to_csv(newpath + "/positions.csv", index=False, header=True)
    lossdf.to_csv(newpath + "/loss.csv", index=False, header=True)


def save_model(policy_net, optimizer, modelpath, total_tstep, model_save_interval):
    """Save the most recet neural network model"""
    policy_net_state = {
        "state_dict": policy_net.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(policy_net_state, modelpath + "/model_recent.h5")

    """Save neural network model at a specific interval of training steps"""
    s = np.floor(total_tstep / model_save_interval)
    torch.save(policy_net.state_dict(), modelpath + "/model_" + str(int(s + 1)) + ".h5")


def plot_progress(result, loss_array, crash, e, reward_cumu, newpath, arg):
    """Get simaple moving average of reward and loss"""
    reward_sma = get_moving_average(arg[0], result["Cumulative Reward"].tolist())
    loss_sma = get_moving_average(arg[1], loss_array)

    """assign different marker color depending on who the episode ends"""
    if crash == True:
        marker = "ro"
    else:
        marker = "go"

    """Plot reward and loss graphs"""
    if len(loss_array) > 0:
        plt.subplot(2, 1, 1)
        plt.plot(result["Cumulative Reward"], linewidth=0.3)
        plt.plot(reward_sma)
        plt.plot(e, reward_cumu, marker)
        plt.subplot(2, 1, 2)
        plt.plot(loss_array, linewidth=0.1)
        plt.plot(loss_sma)
        if len(loss_sma) > 0:
            plt.ylim(0, max(np.array(loss_sma).flatten()) * 1.1)
        plt.savefig(newpath + "/result.png")
        plt.pause(0.0001)


def numerical_sort(value):
    """Function for sorting file name by number"""
    numbers = re.compile(r"(\d+)")
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def rgb2gray(rgb):
    """Turn RGB type of image into gray-scale image"""
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
