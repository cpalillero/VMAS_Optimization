# main.py

# Import packages
import torch
from train import train

# Agent imports
from Agents.PPO import PPOAgent
from Agents.DDPG import DDPGAgent
from Agents.SAC import SACAgent

if __name__ == "__main__":
    scenario = "reverse_transport" # Scenario name from VMAS Github: https://github.com/proroklab/VectorizedMultiAgentSimulator
    n_agents = 5 # Number of agents in scenario
    max_steps = 500 # Maximum timesteps per environment
    frames_per_batch = 128_000 # Total number of timesteps across all parallel environments before updating policy
    # frames_per_batch / max_steps = number of parallel environments
    n_iters = 25 # Total training iterations/policy updates
    device = ( # Use GPU if available
    torch.device(0)
    if torch.cuda.is_available()
    else torch.device("cpu")
    )
    minibatch_size=8192
    num_epochs=6 

    train(PPOAgent, scenario, n_agents, max_steps, frames_per_batch, n_iters, device, minibatch_size, num_epochs) # Call train() function from train.py script
