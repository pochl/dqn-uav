import os
import shutil
from datetime import datetime

import torch
import torch.optim as optim

from src.libs.dql import DQN, ExperienceReplayer, calculate_reward
from src.libs.controller import Controller
from src.libs.agent import Agent
from src.libs.utils import read_spec, load_yaml, write_yaml
from src.libs.communicator import Communicator
from src.libs.logger import Logger

# =============================================================================
# Read configs
# =============================================================================
config = load_yaml('./jobs/configs/train_config.yaml')
env_params = config['environment_params']
learning_params = config['learning_params']
other_params = config['other_params']

# =============================================================================
# Get Specifications from Unity
# =============================================================================
SpecPath = "" + "../spec.txt"
spec = read_spec(SpecPath)  # , other_params['n_pixel_h'], other_params['n_pixel_v'])
input_dim = (int(spec['dim_v']), int(spec['dim_h']))
input_type = spec['input_type']

config['environment_params'].update(spec)

# =============================================================================
# Start New Training
# =============================================================================

ID = datetime.now().strftime("%d%m%Y%H%M")
exp_root = "../experiments/training"
if not os.path.exists(exp_root):
    os.makedirs(exp_root)
exp_path = exp_root + "/" + spec['input_type'] + "_" + env_params['controller_type'] + "_" + ID
if os.path.exists(exp_path):
    shutil.rmtree(exp_path)
os.makedirs(exp_path)

write_yaml(config, exp_path + '/train_config.yaml')

# Establish connection with Unity and receive initial message.
communicator = Communicator(input_type, input_dim, env_params['dept_est_speed'])
communicator.connect()
initial_data = communicator.receive_data()
communicator.send_data([0, 1])  # Reset the environment
n_observed_state = len(initial_data) - env_params['n_hidden_states']

# Initialisation"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(n_observed_state, learning_params['layers'], env_params['action_space_size']).to(device)
target_net = DQN(n_observed_state, learning_params['layers'], env_params['action_space_size']).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimiser = optim.Adam(
    params=policy_net.parameters(),
    lr=learning_params['alpha_initial'],
)

replayer = ExperienceReplayer(
    learning_params['batch_size'],
    learning_params['replay_epoch'],
    learning_params['gamma'],
    learning_params['target_update']
)

controller = Controller(env_params['action_space_size'],
                        input_dim=input_dim,
                        controller_type=env_params['controller_type'],
                        policy_net=policy_net)

agent = Agent(controller,
              communicator,
              policy_net,
              target_net,
              replayer,
              optimiser,
              learning_params['memory_size'],
              learning_params['alpha_initial'],
              learning_params['alpha_decay'],
              learning_params['alpha_update'],
              learning_params['epsilon_initial'],
              learning_params['epsilon_decay'],
              learning_params['epsilon_min']
              )

recorder = Logger(exp_path=exp_path)

# =============================================================================
# Begin the Training
# =============================================================================
for e in range(learning_params['n_episodes']):

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

        agent.replay_experience(next_state[-n_observed_state:], reward, crash)

        # Update the record
        recorder.update(reward, action=agent._action, position=next_state[2:4], crash=crash)

        # Transit to the next state
        state = next_state

    # Reset the environment
    communicator.send_data([0, 1])

    # Save record
    recorder.save_record(agent.replayer._loss_record)

    # Save model
    recorder.save_model(policy_net, optimiser, agent._total_tstep, other_params['model_save_interval'])

    # Plot progress"""
    recorder.plot(e, agent.replayer._loss_record, other_params['sma_period_reward'], other_params['sma_period_loss'])

    agent.update(e)

communicator.disconnect()
