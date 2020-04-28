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
import socket
import numpy as np
from collections import namedtuple
import pandas as pd
from datetime import datetime
import os
import shutil
from pathlib import Path
import torch
import torch.optim as optim
from transfer_data import transfer_data
from DQL import update_learning_rate, ReplayMemory,\
                 ExperienceReplay, Agent, DQN
from OtherFunc import read_spec, plot_progress, save_result, \
                        save_model, choose_action_HC
# =============================================================================

"""Choose whether to start a new trainng or continue from the previous one"""  
StartNewSim = True

"""Choose the type of controller. 
'RL' => Reinforcement Learning controller,
'HC' => Hand controller (Breaitenberg controller) """
Controller = 'RL'

# =============================================================================
# Learning Parameters 
# =============================================================================
gamma = 0.95          #Reward discount factor
epsilon_initial = 1   #Initial value of epsilon
epsilon_min = 0.01    #Minimum possible epsilon (the value it will anneal to)
epsilon_decay = 0.003 #Epsilon decay factor. (Higher -> faster decay rate)
alpha_initial = 0.001 #Initial value of learning rate
alpha_decay = 0.5     #Learning rate decay factor (Hgiher -> slower decay)
batch_size = 64       #The size of the batch of sampled experiences
target_update =3000   #Interval to update target network
lr_update = 30000     #Interval to step down learning rate
memory_size = 300000  #Capacity of replay memory  
layers = [128,128]    #Number of nodes of 1st and 2nd hidden layers of NN
max_env_steps = 1000  #Maximum training steps of each episode
n_episodes = 10000    #Maximum number of episodes for one training  
replay_epoch = 1      #Number of epoch for an experience replay


# =============================================================================
# Other Parameters
# =============================================================================
NumPixelHor = 10       #No. of horizontal pixels to resize the image to
NumPixelVer = 10        #No. of vertical pixels to resize the image to
DeptEstSpeed = 0.1      #Time delay for depth estimation algorithm
truestate = 3           #No. of elements in true state excluding LiDAR/image
action_space_size = 3   #Size of action space
model_save_interval = 10000 #Interval to save NN model
sma_period_reward = 100 #period to caculate moving average of reward graph
sma_period_loss = 1000  #period to caculate moving average of loss graph
# =============================================================================
# Get Specifications from Unity 
# =============================================================================
mainpath = str(Path(__file__).parents[1])
SpecPath = mainpath + '/spec.txt'
spec = read_spec(SpecPath,NumPixelHor,NumPixelVer)
InputDim = [spec.values[0,2], spec.values[0,1]]
InputType = spec.values[0,0]


# =============================================================================
# Start New Training or Continue from what was left off
# =============================================================================
if StartNewSim == True:
    """Initialise parameters, result tanks, and replay memory"""
    epsilon = epsilon_initial
    alpha = alpha_initial
    memory = ReplayMemory(memory_size)
    loss_array = []
    positions = []
    result = pd.DataFrame(columns=['Cumulative Reward', \
                'Time Step', 'Crash','Straight', 'Left Turn', 'Right Turn'])
    e_c = 0
    total_tstep = 0
    image_old = np.ones([NumPixelVer,NumPixelHor])
    Experience = namedtuple('Experience',
                       ('state','action','next_state','reward', 'crash'))
    
    
    """Create new file path for new experiment""" 
    ID = datetime.now().strftime("%d%m%Y%H%M")  
    newpath = mainpath + '/History'
    if not os.path.exists(newpath):
        os.makedirs(newpath)   
    newpath = newpath + '/' + spec.values[0,0] + '_' + Controller + '_' +ID
    if os.path.exists(newpath):
        shutil.rmtree(newpath)
    os.makedirs(newpath)
    modelpath = newpath + '/model'
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)
    
    """Create text file containing learning parameters for this experiment""" 
    inputs_list = ['gamma = ' + str(gamma) + '\n',\
                   'epsilon_initial = ' + str(epsilon_initial) + '\n',\
                   'epsilon_min = ' + str(epsilon_min) + '\n',\
                   'epsilon_decay = ' + str(epsilon_decay) + '\n',\
                   'alpha_initial = ' + str(alpha_initial) + '\n',\
                   'alpha_decay = ' + str(alpha_decay) + '\n',\
                   'target_update = ' + str(target_update) + '\n',\
                   'lr_update = ' + str(lr_update) + '\n',\
                   'memory_size = ' + str(memory_size) + '\n',\
                   'batch_size = ' + str(batch_size) + '\n',\
                   'NN = ' + str(layers[0]) + ',' + str(layers[1]) + '\n',\
                   'Outcome:']
    file = open(newpath + '/info.txt', 'w')
    file.writelines(inputs_list)
    file.close()
    spec.to_csv(newpath + '/spec.csv', index=None, sep=',', mode='a')
    
    """Initialise neural network model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(InputDim,layers, truestate, action_space_size).to(device)
    target_net = DQN(InputDim,layers, truestate, action_space_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(params = policy_net.parameters(), lr = alpha, )
    
elif StartNewSim == False: 
    """Get values and models from what was left off"""
    e_c = len(result)
    checkpoint = torch.load(modelpath + '/model_recent.h5')
    policy_net = DQN(InputDim,layers, truestate, action_space_size).to(device)
    target_net = DQN(InputDim,layers, truestate, action_space_size).to(device)
    policy_net.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(params = policy_net.parameters(), lr = alpha)
    optimizer.load_state_dict(checkpoint['optimizer'])
    target_net.load_state_dict(policy_net.state_dict())

"""Create connection with Unity"""
host, port = "127.0.0.1" , 25001 #Must be identical to the ones in Unity code
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host,port)) 

"""initialise classes"""
transfer_data = transfer_data(sock, InputType, 
                              InputDim, DeptEstSpeed, truestate)
ExperienceReplay = ExperienceReplay(Experience, batch_size,
                                    replay_epoch, gamma, target_update)
Agent = Agent(action_space_size)

# =============================================================================
# Begin the Training 
# =============================================================================
for e in range(e_c,n_episodes):

    """Initialise episodic results"""
    reward_cumu = 0
    tstep = 0
    act = [0,0,0]
    loss_tmt = []
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
                    torch.tensor([state]), epsilon, policy_net, image)
        else:
            action = choose_action_HC(image, InputDim)
        transfer_data.SendData([action,0])
        
        """Get next state and other information from Unity"""
        data, next_state, image, rem = transfer_data.ReceiveData(image_old)
        crash = data[0]         
        reward = Agent.get_reward(data)
        
        """Store experience in replay memory"""
        memory.push(Experience(torch.tensor([state]), torch.tensor([action]), 
                        torch.tensor([next_state]), torch.tensor([reward]), 
                        torch.tensor([crash])), rem)
        
        """Perform experience replay & update target network by the schedule"""
        ExperienceReplay.replay(memory, policy_net, target_net, optimizer, 
                                loss_tmt, total_tstep)

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
    loss_array += loss_tmt
    positions += positions_tmt
    result = result.append({'Cumulative Reward': reward_cumu.item(),\
                        'Time Step': tstep,\
                        'Crash': int(crash),\
                        'Straight': act[0],\
                        'Left Turn': act[1],\
                        'Right Turn': act[2]}, ignore_index=True)
    
    """Save results and model"""
    save_result(positions, loss_array, result, newpath)
    save_model(policy_net, optimizer, modelpath, 
               total_tstep, model_save_interval)
    
    """Plot progress"""
    plot_progress(result, loss_array, crash, e, reward_cumu, 
                  newpath, [sma_period_reward, sma_period_loss])
    
    """Update learning rate"""
    update_learning_rate(alpha_initial, alpha_decay, 
                         total_tstep, lr_update, optimizer)

    """Update epsilon"""
    epsilon = max(epsilon_min, epsilon_initial * ((1 - epsilon_decay) ** e))
    
    
sock.close()