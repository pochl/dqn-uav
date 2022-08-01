#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 01:36:46 2019

@author: thepoch
"""
from __future__ import print_function
import socket
import numpy as np
import pandas as pd
import os, glob
from pathlib import Path
import torch
import re
from transfer_data import transfer_data
from DQL import Agent, DQN
from OtherFunc import choose_action_HC

def numericalSort(value):
    """Function for sorting file name by number"""
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

"""Choose the type of controller. 
'RL' => Reinforcement Learning controller,
'HC' => Hand controller (Breaitenberg controller) """   
Controller = 'RL'
# =============================================================================
# Testing Parameters
# =============================================================================
Folder_Name = '**Visual_RL_210420200111' #File name of the experiment folder 
max_env_steps = 220    #Max episode steps
n_episodes = 20       #Max number of episode for each test
DeptEstSpeed = 0.1
epsilon = 0             #Set to 0 for testing
action_space_size = 3
truestate = 3



# =============================================================================
# Initialsing
# =============================================================================
"""Get paths"""
mainpath = str(Path(__file__).parents[1])
path = mainpath + '/History/' + Folder_Name
modelpath = path + '/model'

"""Read info and spec of the experiment"""
info = pd.read_csv(path + '/info.txt', sep=" ", header = None)
NN = info.loc[info[0] == 'NN'].values[0,2].split(',')
layers = [int(NN[0]), int(NN[1])]
spec = pd.read_csv(path + '/spec.csv', sep=",")
InputDim = [spec.values[0,2], spec.values[0,1]]
InputType = spec.values[0,0]

"""Initialise parameters, result tanks, and replay memory"""
epsilon = 0

total_tstep = 0
image_old = np.ones(InputDim)

"""Create new file path in current experiment's folder for test result""" 
testpath = path + '/test'
if not os.path.exists(testpath):
    os.makedirs(testpath)

"""Initialise neural network model"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(InputDim,layers,truestate,action_space_size).to(device)

"""Create connection with Unity"""
host, port = "127.0.0.1" , 25001 #Must be identical to the ones in Unity code
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host,port)) 

"""initialise classes"""
transfer_data = transfer_data(sock, InputType, 
                              InputDim, DeptEstSpeed, truestate)
Agent = Agent(action_space_size)

# =============================================================================
# Begin the Testing 
# =============================================================================
for filename in sorted(glob.glob(os.path.join(modelpath, '*.h5')), 
                       key=numericalSort):
    """Iterate the models in the model folder to test one by one"""
    file = filename.split("/")[-1]
    file = file[:-3]
    print(file)
    policy_net = DQN(InputDim,layers,truestate, action_space_size).to(device)
    policy_net.load_state_dict(torch.load(filename))
    policy_net.eval() 
    
    """Initialise result tanks for each model"""
    positions = []
    result = pd.DataFrame(columns=['Cumulative Reward', 'Time Step', 
                                   'Crash','Straight', 'Left Turn', 
                                   'Right Turn'])  
    for e in range(0,n_episodes):
        print(e)        
        """Initialise episodic results"""
        reward_cumu = 0
        tstep = 0
        act = [0,0,0]
        positions_tmt = []
        
        """Initialise the simulation"""   
        data_received = transfer_data.ReceiveData(image_old)                               
        transfer_data.SendData([0,1]) #Reset the environment
    
        """Get first set of state"""
        data, state, image, rem = transfer_data.ReceiveData(image_old)
        crash = data[0]

        while not crash and tstep < max_env_steps:
            """Choose and send action to Unity"""
            if Controller == "RL":
                action = Agent.choose_action_RL(
                        torch.tensor([state]), epsilon, 
                        policy_net, image)
            else:
                action = choose_action_HC(image, InputDim)
            transfer_data.SendData([action,0])
            
            """Get next state and other information from Unity"""
            data, next_state, image, rem = transfer_data.ReceiveData(image_old)
            crash = data[0]         
            reward = Agent.get_reward(data)      
    
            """Trainsition to the next state"""
            state = next_state
            image_old = image   #Save previous image in case error occurs 
                                #during image dceoding in the next loop
            
            """Append or cumulate results"""
            reward_cumu += reward
            tstep += 1
            act[action] += 1
            total_tstep += 1
            positions_tmt.append(data[2:4])

        """Reset the environment"""
        transfer_data.SendData([0,1])

        """Store episodic results in result tank"""
        positions += positions_tmt
        result = result.append({'Cumulative Reward': reward_cumu.item(),\
                            'Time Step': tstep,\
                            'Crash': int(crash),\
                            'Straight': act[0],\
                            'Left Turn': act[1],\
                            'Right Turn': act[2]}, ignore_index=True)
    
        """Save results"""
        positionsdf = pd.DataFrame.from_records(positions, columns = ['x','z'])
        result.to_csv(testpath + '/result_test_' + \
                      file + '.csv', index = False, header=True)
        positionsdf.to_csv(testpath + '/positions_test_' + \
                           file + '.csv',index = False, header=True)

sock.close()
    
   
