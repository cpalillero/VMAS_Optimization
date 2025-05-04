# train.py

# Import packages
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchrl.collectors import SyncDataCollector
from make_env import make_vmas_env

def train(agent_class, scenario, n_agents, max_steps, frames_per_batch, n_iters, device, minibatch_size, num_epochs): # Environment training function
    # agent_class: Agent class type
    # scenario: VMAS scenario name
    # n_agents: Number of agents in scenario
    # max_steps: Maximum timesteps per environment
    # frames_per_batch: Total number of timesteps across all parallel environments before updating policy
    # n_iters: Total training iterations/policy updates
    # device: CPU or GPU used
    # minibatch_size: Size of minibatches during training
    # num_epochs: Number of epochs per iteration
    env = make_vmas_env(scenario, frames_per_batch, max_steps, n_agents, device) # Create environment object using passed in parameters
    agent = agent_class(env=env, device=device) # Create agent using environment and device

    collector = SyncDataCollector( # Synchronous data collector for parallel environments
        env, # Use environment object
        agent.policy, # Use current agent policy
        frames_per_batch=frames_per_batch, # Use passed in frames_per_batch value
        total_frames=frames_per_batch * n_iters, # Calculate total frames across all iterations
        device=device, # CPU or GPU used
    )

    pbar = tqdm(total=n_iters) # Sets up progress bar in output
    rewards = [] # Creates empty reward list

    for data in collector: # For each iteration: collector gets a TensorDict batch of transitions (observations, actions, rewards, dones, etc.)
        
        # Expand done/terminated flags to match episode reward tensor shape
        data.set(("next", "agents", "done"), data.get(("next", "done")).unsqueeze(-1).expand_as(data.get(("next", "agents", "episode_reward"))))
        data.set(("next", "agents", "terminated"), data.get(("next", "terminated")).unsqueeze(-1).expand_as(data.get(("next", "agents", "episode_reward"))))

        # Let agent handle its own preprocessing
        if hasattr(agent, "preprocess_batch"):
            agent.preprocess_batch(data)


        data_view = data.reshape(-1) # Flatten multi-dimensional batch into 1D batch for training
        agent.update(data_view, minibatch_size=minibatch_size, num_epochs=num_epochs, max_grad_norm=0.5) # Update agent's policy and critics with flattened batch, minibatch size, number of epochs, and gradient clipping

        collector.update_policy_weights_() # Copy most recent agent policy weights into collector

        reward = data.get(("next", "agents", "episode_reward"))[data.get(("next", "agents", "done"))].mean().item() # Compute mean episode reward per agent for all completed episodes this iteration
        rewards.append(reward) # Update reward list
        pbar.set_description(f"Reward = {reward:.2f}") # Output reward value
        pbar.update() # Update output

    plt.plot(rewards)
    plt.title(f"{scenario} Agent Reward")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Episode Reward")
    plt.show()

    with torch.no_grad():
        env.rollout(
            max_steps=max_steps,
            policy=agent.policy,
            callback=lambda env, _: env.render(),
            auto_cast_to_device=True,
            break_when_any_done=False,
        )

    return rewards  # List of mean episode rewards per iteration
