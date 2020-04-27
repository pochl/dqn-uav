#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:48:17 2020

@author: thepoch
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import numpy as np
import scipy.interpolate
import glob, os
import re
from scipy.ndimage.filters import gaussian_filter1d

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()
    
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# =============================================================================
# Read Results from Files
# =============================================================================
Folder_Name = '*Visual_HC_230420201646'
Folder_Name = '*LiDAR_HC_230420202102'

path = '/Users/thepoch/Desktop/3rdYearProject2/History/' + Folder_Name
results_all = pd.read_csv(path + '/result.csv', sep=",")
positions_all = pd.read_csv(path + '/positions.csv', sep=",")
loss_all = pd.read_csv(path + '/loss.csv', sep=",")

# =============================================================================
# Extract Results within Interested Period  
# =============================================================================
ts_req = 300000
ts_cumsum_all = np.cumsum(results_all['Time Step'])
ep_req = sum(ts_cumsum_all < ts_req) + 1

loss = loss_all[:ts_req]
results = results_all[:ep_req]
positions = positions_all[:ts_req]

# =============================================================================
# Plot Trajectories 
# =============================================================================

trajectories = []
c = 0
for ts in results['Time Step']:
    trajectories.append(positions[int(c):int(c)+int(ts)])
    c = int(c)+int(ts)
    
for i in range(len(trajectories)):
    a = np.sign(np.diff(trajectories[i]['z'])) < 0
    if len(trajectories[i]) != 1000:
        color = 'black'
        lw = 1
        plt.plot(trajectories[i]['x'].values[-1], trajectories[i]['z'].values[-1], 'ro')
    else:
        color = 'C0' 
        lw = 0.3
    plt.plot(trajectories[i]['x'],trajectories[i]['z'],
             color = color, alpha = 0.5, linewidth=lw)
plt.title('LiDAR with Braitenberg Controller')
plt.xlim(-200, 200)
plt.ylim(0, 300)
plt.xlabel('x [m]')
plt.ylabel('z [m]')
plt.savefig(path + '/trajectories2.png')


# =============================================================================
# Plot Episode Rewards vs Training Steps
# =============================================================================
RewardTS = [[],[]]
ts_cumsum = np.cumsum(results['Time Step'])

period = 1#int(0.01*len(ts_cumsum))

for p in range(int(np.ceil(ep_req/period))):
    #RewardTS += list(np.ones(int(results['Time Step'][ep])) * results['Cumulative Reward'][ep])
    #print(p)
    RewardTS[0].append(np.mean(ts_cumsum[(p*period):((p+1)*period)]))
    RewardTS[1].append(np.mean(results['Cumulative Reward'][(p*period):((p+1)*period)]))
RewardTS_D = RewardTS

plt.plot(RewardTS[0],ysmoothed_D)
plt.plot(RewardTS[0],RewardTS[1])  

ysmoothed_L = gaussian_filter1d(RewardTS_L[1], sigma=15)
ysmoothed_D = gaussian_filter1d(RewardTS_D[1], sigma=60)
ysmoothed_D2 = gaussian_filter1d(RewardTS_D2[1], sigma=30)

plt.plot( np.array(RewardTS_L[0])/1000,  ysmoothed_L/1000, label='LiDAR')
plt.plot( np.array(RewardTS_D[0])/1000, ysmoothed_D/220, label='Depth Estimation (0.1s delay)') 
plt.plot( np.array(RewardTS_D2[0])/1000,  ysmoothed_D2/500, label='Depth Estimation (0.03s delay)')
plt.ylabel('Normalized Episode Rewards') 
plt.xlabel('Training Steps (x1000)')  
plt.ylim(0,1)  
plt.legend(loc='lower right')  
    
    
# =============================================================================
# Plot Loss
# =============================================================================    
loss_D2 = get_moving_average(1000,loss['loss'])


plt.subplot(211) 
plt.plot(np.array(range(len(loss_D)))/1000,loss_L, label='LiDAR', color = 'C0')
plt.legend(loc='lower right')
plt.ylabel('Loss') 
plt.subplot(211)
plt.plot(np.array(range(len(loss_D)))/1000,loss_D, label='Depth Estimation (0.1s delay)', color = 'C1')
plt.legend(loc='lower right')
plt.ylabel('Loss')
plt.subplot(212)
plt.plot(np.array(range(len(loss_D2)))/1000,loss_D2, label='Depth Estimation (0.03s delay)', color = 'C2')
plt.ylabel('Loss') 
plt.legend(loc='lower right') 
plt.xlabel('Training Steps (x1000)')    

# =============================================================================
# Plot Turns
# =============================================================================  
ts_cumsum = np.cumsum(results['Time Step'])
turns = [[],[],[]]

turns[0] = results['Straight'] / (results['Straight'] + results['Left Turn'] + results['Right Turn'])
turns[1] = results['Left Turn'] / (results['Straight'] + results['Left Turn'] + results['Right Turn'])
turns[2] = results['Right Turn'] / (results['Straight'] + results['Left Turn'] + results['Right Turn'])

turns[0] = gaussian_filter1d(turns[0], sigma=15)
turns[1] = gaussian_filter1d(turns[1], sigma=15)
turns[2] = gaussian_filter1d(turns[2], sigma=15)

plt.plot(ts_cumsum/1000, turns[0]*100, label = 'Straight')
plt.plot(ts_cumsum/1000, turns[1]*100, label = 'Left Turn')
plt.plot(ts_cumsum/1000, turns[2]*100, label = 'Right Turn') 
plt.ylabel('%') 
plt.xlabel('Training Steps (x1000)') 
plt.ylim(0,100)
plt.title('LiDAR')
plt.legend(loc='best') 

# =============================================================================
# Process Test Result
# =============================================================================  
Folder_Name = '**LiDAR_RL_200420201534'
Folder_Name = '**Visual_RL_210420200111'
path = '/Users/thepoch/Desktop/3rdYearProject2/History/' + Folder_Name
path_result =  path + '/test/result'
path_position = path + '/test/positions'

result_test =  pd.DataFrame(columns=['Cumulative Reward', 'Time Step', 'Crash','Straight', 'Left Turn', 'Right Turn'])
f = 1    

for filename in sorted(glob.glob(os.path.join(path_result, '*.csv')), key=numericalSort):
        file = filename.split("/")[-1]
        file = file[:-4]
        print(file)
        result = pd.read_csv(filename)
        mean = result.mean(axis = 0)
        result_test = result_test.append({'Cumulative Reward': mean[0],\
                                'Time Step': mean[1],\
                                'Crash': mean[2],\
                                'Straight': mean[3]/sum(mean[3:6]),\
                                'Left Turn': mean[4]/sum(mean[3:6]),\
                                'Right Turn': mean[5]/sum(mean[3:6])}, ignore_index=True)

    
        positions = pd.read_csv(path_position + '/positions_test_model_' + str(f) + '.csv', sep=",")
#        positions['x'] =  -positions['x']
#        positions['z'] = 750 - positions['z']
#        positions.to_csv(path + '/test/positions/positions_test_model_' + str(f) + '.csv',index = False, header=True)

        
        
        trajectories = []
        c = 0
        for ts in result['Time Step']:
            trajectories.append(positions[int(c):int(c)+int(ts)])
            c = int(c)+int(ts)
            
        for i in range(len(trajectories)):
            plt.plot(trajectories[i]['x'],(trajectories[i]['z'])*1.2,
                     color = 'C0', alpha = 0.5, linewidth=0.3)
        
        plt.title('LiDAR Model ' + str(f) + ': At ' + str(f*10) + 'k training step')
        plt.xlim(-200, 200)
        plt.ylim(0, 300)
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.savefig(path + '/test/trajectories_test_' + str(f) + '.png')
        plt.pause(0.001)
        
        
        
        f += 1

fig,ax = plt.subplots()
ax.plot((np.array(range(len(result_test)))+1),result_test['Cumulative Reward']/220, 'o-', color = 'C0')
ax.set_xlabel('Model', fontsize =14)
ax.set_ylabel('Normalized Episode Rewards', color = 'C0', fontsize =14)
ax.set_ylim(top = 1.1)
ax2 = ax.twinx()
ax2.plot((np.array(range(len(result_test)))+1),result_test['Crash']*100, 'o-', color = 'C1')
ax2.set_ylabel('Crash [%]', color = 'C1', fontsize =14)
ax2.set_ylim(0,110)
plt.axvline(x = 26, color = 'red') 
plt.title('Depth Estimation (0.1s delay)', fontsize = 16)
plt.show()




Folder_Name = '**LiDAR_RL_200420201534'
#Folder_Name = '*Visual_RL_210420200111'

path = '/Users/thepoch/Desktop/3rdYearProject2/History/' + Folder_Name + '/test/positions'

result_test =  pd.DataFrame(columns=['Cumulative Reward', 'Time Step', 'Crash','Straight', 'Left Turn', 'Right Turn'])
    

for filename in sorted(glob.glob(os.path.join(path, '*.csv')), key=numericalSort):
        file = filename.split("/")[-1]
        file = file[:-4]
        print(file)
        
        trajectories = []
        c = 0
        for ts in results['Time Step']:
            trajectories.append(positions[int(c):int(c)+int(ts)])
            c = int(c)+int(ts)
            
        for i in range(len(trajectories)):
            plt.plot(trajectories[i]['x'],trajectories[i]['z'],
                     color = 'blue', alpha = 0.5, linewidth=0.1)
        plt.xlim(-200, 200)
        plt.ylim(0, 300)
        #plt.savefig(path + '/trajectories.png')
        



    
            