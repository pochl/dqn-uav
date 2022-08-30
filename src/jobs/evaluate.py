import glob
import os

import torch

from src.libs.dql import DQN, calculate_reward
from src.libs.agent import Agent
from src.libs.controller import Controller
from src.libs.communicator import Communicator
from src.libs.utils import load_yaml, numerical_sort
from src.libs.logger import Logger

# =============================================================================
# Read configs
# =============================================================================
config = load_yaml('./jobs/configs/test_config.yaml')

# =============================================================================
# Initialsing
# =============================================================================
exp_path = "../experiments/training/" + config['experiment']
model_folder = exp_path + "/models"
training_config = load_yaml(exp_path + '/train_config.yaml')
env_params = training_config['environment_params']
learning_params = training_config['learning_params']

# Create new file path in current experiment's folder for test result
testpath = "../experiments/evaluation/" + config['experiment']
if not os.path.exists(testpath):
    os.makedirs(testpath)

# Read config of the experiment
layers = learning_params['layers']
input_type = env_params['input_type']
input_dim = (env_params['dim_v'], env_params['dim_h'])

# Establish connection with Unity and receive initial message.
communicator = Communicator(input_type, input_dim, env_params['dept_est_speed'])
communicator.connect()
initial_data = communicator.receive_data()
communicator.send_data([0, 1])  # Reset the environment
n_observed_state = len(initial_data) - env_params['n_hidden_states']

# Initialisation"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(n_observed_state, learning_params['layers'], env_params['action_space_size']).to(device)

controller = Controller(env_params['action_space_size'],
                        input_dim=input_dim,
                        controller_type=env_params['controller_type'],
                        policy_net=policy_net)

agent = Agent(controller, communicator, policy_net)

recorder = Logger(exp_path=exp_path)

# =============================================================================
# Begin the Testing
# =============================================================================
for filename in sorted(glob.glob(os.path.join(model_folder, "*.h5")), key=numerical_sort):
    """Iterate the models in the model folder to test one by one"""
    file = filename.split("/")[-1]
    file = file[:-3]
    print(file)
    policy_net.load_state_dict(torch.load(filename))
    policy_net.eval()

    for e in range(0, learning_params['n_episodes']):

        """Initialise episodic results"""
        recorder.reset()

        """Get first set of state"""
        state = communicator.receive_data()
        crash = state[0]

        while not crash and recorder.tstep < learning_params['max_env_steps']:

            agent.step(state[-n_observed_state:])

            # Get next state and other information from Unity
            next_state = communicator.receive_data()
            crash = next_state[0]
            reward = calculate_reward(next_state)

            # Update the record
            recorder.update(reward, action=agent._action, position=next_state[2:4], crash=crash)

            # Transit to the next state
            state = next_state

        # Reset the environment
        communicator.send_data([0, 1])

        # Save record
        recorder.save_record()

communicator.disconnect()
