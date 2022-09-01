# Fixed-Wing UAV Obstacles Avoidance with Deep Q-Learning
This project is my 3rd Year dissertation for BEng Mechanical Engineering at the University of Manchester. The project started at the beginning of Fall 2019 semester and finished at the end of Spring 2020 semester under the supervision of Prof. William Crowther. The full project report is in the file `docs/Full_Report.pdf`. The video showing the learning progress and final result of the agent is available at: https://youtu.be/DPaUbFxIfHM

<img src=docs/figures/env1.png width = 400>     <img src=docs/figures/env2.png width = 400>


## Project Description
This project implements Deep Q-Learning into one of the most heavily researched tasks in automation, obstacles avoidance. The environment and the UAV itself are simulated in Unity3D, while the DQL algorithm is written Python. The action of the UAV is constrained into only turning left, right, and going straight. That means its movement is limited to a 2D horizontal plane, very similar to a car. However, the UAV also rolls while it's turning, just like a normal plane that needs to roll in order to turn. Thus, the environment is still in 3D. 

The goal of the agent is to fly in the forest of obstacles for a certain period of time. To prevent the agent from circling around, target point is introduced for the agent to fly to. This target point is NOT for the agent to reach, but only to guide the UAV into one particular direction. 

The agent will receive a reward of +1 for every time step that it gets closer to the target point. However, if it flies further away from the target point at any time step, it will be given a reward of -1. If the agent crashes, it will get -1 reward, regardless of its flight direction. The reason for such a simple reward is to increase the stability in learning of the agent. This is called "reward clipping".

There are 2 types of input that are used in this project:
 
1. LiDAR
  - simulated by using raycasts in Unity
  - produce only 2D input
  - wide field of view [-90° , 90°] horizontally
        
2. Depth image
  - mimicking the use of depth estimation algorithm with monocular camera
  - simulated using depth rendering in Unity
  - produce 3D input 
  - narrow field of view [-30° , 30°] horizontally & vertically
  - 0.1s time delay at each time step (mimic the slow computational time of depth estimation algorithm)
  
As you can see, each input has pros and cons. This project will conduct the experiment with each input and assess the effect of their characteristics on the agent's performance in avoiding obstacles. 

### Installation Steps

1. Clone the repo

#### Simulation Setup:
1. Download Unity3D (version 2018.2.9f1). This will be the software that used to run the simulation
2. From the repo, add the folder 'UAV_Environment_Unity' into the project in Unity Hub.

#### Python Scripts Setup:
For the python scripts, this repo uses `poetry` to manage the packages and dependencies.
1. [Optional] Create your choice of virtual environment (e.g. `conda`, `virtualenv`, etc.)
2. From the project directory in terminal, run `pip install poetry`.
3. Run `poetry install`. This will install all the packages listed on the `poetry.lock` file.

#### Running the experiment.
1. To train the model, opens the unity project and click the `run` button at the top panel. 
2. Run the python script `src/jobs/train.py`.
3. The configurations can be set in the config file `src/jobs/configs/train_config.yaml`.

#### Evaluate the models.
1. Opens the unity project and click the `run` button at the top panel. 
2. Run the python script `src/jobs/evaluate.py`.
3. The configurations can be set in the config file `src/jobs/configs/eval_config.yaml`.
