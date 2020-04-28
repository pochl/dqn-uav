#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 19:23:54 2020

@author: thepoch
"""
from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def update_learning_rate(alpha_initial, alpha_decay, 
                         total_tstep, lr_update, optimizer):
    """Update learning rate"""
    alpha = alpha_initial * ((alpha_decay)**np.floor(total_tstep/lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha

class Agent():  
    def __init__(self, Controller, action_space_size, InputDim):
        self.Controller = Controller
        self.action_space_size = action_space_size
        self.InputDim = InputDim
       
    def get_reward(self, data):
        """Calculate reward at current time step"""
        distdiff = data[3]
        crash = bool(data[0])  
        reward = np.sign(distdiff)*(1-crash) - crash
        return reward
    
    def choose_action_RL(self, state, epsilon, policy_net, image):
        """e-greedy strategy"""
        if np.random.random() <= epsilon:
            action = random.randrange(self.action_space_size)
        else:
            with torch.no_grad():
                predict = policy_net(state)
                action = predict.argmax(dim=1).item()   
        return int(action)

class DQN(nn.Module):
    def __init__(self, InputDim,layers):
        super().__init__()      
        """create nn layers"""
        self.fc1 = nn.Linear(in_features = np.prod(InputDim) + 3, 
                             out_features = layers[0])
        self.fc2 = nn.Linear(in_features = layers[0], out_features = layers[1])
        self.out = nn.Linear(in_features = layers[1], out_features = 3)

    def forward(self, t):
        """feed forward"""
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        
    def push(self, experience, rem):
        if rem:
            if len(self.memory) < self.capacity:
                """add new experience"""
                self.memory.append(experience)
            else:
                """push the oldest one out if memory is full"""
                self.memory[self.push_count % self.capacity] = experience
            self.push_count += 1
    
    def sample(self, batch_size):
        """sample a batch of experience"""
        return random.sample(self.memory, batch_size)
    
    def can_provide_sample(self, batch_size):
        """Check if there's enough experiences in memory to be sampled"""
        return len(self.memory) >= batch_size

class ExperienceReplay():
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __init__(self, Experience, batch_size, 
                 replay_epoch, gamma, target_update):
        self.Experience = Experience
        self.batch_size = batch_size
        self.replay_epoch = replay_epoch
        self.gamma = gamma
        self.target_update = target_update  
        
    def extract_tensors(self, experiences):
        batch = self.Experience(*zip(*experiences))
        t1 = torch.cat(batch.state)
        t2 = torch.cat(batch.action)
        t3 = torch.cat(batch.reward)
        t4 = torch.cat(batch.next_state)
        t5 = torch.cat(batch.crash)
        return (t1,t2,t3,t4,t5)

    @staticmethod
    def Q_current(policy_net, states, actions):
        """Get Q(s_t,a_t)"""
        return policy_net(states).gather(dim = 1, index=actions.unsqueeze(-1))
    
    @staticmethod
    def Q_next(target_net, next_states, crashes):
        """Get max(Q(s_t+1,a_t+1))"""
        final_state_locations = crashes.eq(1).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_state = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(ExperienceReplay.device)
        values[non_final_state_locations] = \
                            target_net(non_final_state).max(dim=1)[0].detach()
        return values

    def replay(self, memory, policy_net, target_net, 
               optimizer, loss_tmt, total_tstep):
        if memory.can_provide_sample(self.batch_size):
            
            """Update target network if it's time for update"""
            if total_tstep % self.target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
                print("target network updated")
                
            """Perform experience replay"""
            for epoch in range(self.replay_epoch):
                experiences = memory.sample(self.batch_size)#Sample experiences
                states,actions,rewards,next_states,crashes = \
                            ExperienceReplay.extract_tensors(self, experiences)
                current_q_values = ExperienceReplay.Q_current(
                                        policy_net, states, actions)
                next_q_values = ExperienceReplay.Q_next(target_net, 
                                                        next_states, crashes)
                target_q_values = (next_q_values * self.gamma) + rewards
                
                """Get Huber loss and perform gradient descend"""
                loss = F.smooth_l1_loss(current_q_values, 
                                        target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_tmt.append(loss.item())