# make_env.py

# Import packages
import torch
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv

def make_vmas_env(scenario_name, frames_per_batch, max_steps, n_agents, device): # Create VMAS environment
    num_vmas_envs = frames_per_batch // max_steps # Number of parallel environments used during training

    base_env = VmasEnv( # Base VMAS environment
        scenario=scenario_name, # VMAS scenario name
        num_envs=num_vmas_envs, # Number of parallel environments used during training
        continuous_actions=True, # Toggle for continuous action space
        max_steps=max_steps, # Maximum timesteps per environment
        device=device, # CPU or GPU used
        n_agents=n_agents, # Number of agents in scenario
        render_mode="human", # Allows visualization after training is completed
    )

    env = TransformedEnv( # Additional transform on top of environment
        base_env, # Use base environment from before
        RewardSum(in_keys=[base_env.reward_key], out_keys=[("agents", "episode_reward")]), # Sum the rewards over the episode for each agent, take rewards from base_env.reward_key, then output to a new key "agents", "episode_reward"
    )

    return env # Return transformed environment for training
