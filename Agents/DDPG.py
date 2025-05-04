# DDPG.py

# Import packages
import copy
import torch
import torch.nn.functional as F
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.modules import MultiAgentMLP


class TanhRescale(nn.Module):
    # Take raw output, apply tanh() in order to bound [-1,1], then rescale result into environment's action range [low, high]
    # Ensures final action output is valid in the environment 
    # DDPG needs this since actions are continuous + bounded
    def __init__(self, low, high):
        super().__init__()
        low = torch.as_tensor(low)
        high = torch.as_tensor(high)
        self.register_buffer("scale", (high - low) / 2.0)
        self.register_buffer("bias",  (high + low) / 2.0)

    def forward(self, x):
        return torch.tanh(x) * self.scale + self.bias


class DDPGAgent: # DDPG Agent class
    # Initialize agent with PPO parameters
    def __init__(
        self,
        env,
        device,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99, # Discount factor
        tau: float = 0.01, # Soft target update rate (how quickly networks update towards learned networks)
        buffer_size: int = 100_000,
        batch_size: int = 1024,
    ):
        #Set up device, environment, number of agents, gamma, tau
        self.env      = env
        self.device   = device
        self.gamma    = gamma
        self.tau      = tau
        self.n_agents = env.n_agents

        # env.full_action_spec_unbatched: Dictionary mapping keys (agents, action, etc.) to action space specification
        # env.action_key: Action space specification
        # Rescaling actor's output to be within [low, high] bounds; regardless of environment being used
        low  = env.full_action_spec_unbatched[env.action_key].space.low
        high = env.full_action_spec_unbatched[env.action_key].space.high

        # Set up observation and action dimensions to size networks
        obs_dim = env.observation_spec["agents","observation"].shape[-1]
        act_dim = env.action_spec.shape[-1]

        # Actor core
        self._actor_core = MultiAgentMLP( 
            n_agent_inputs=obs_dim, # Number of observations per agent
            n_agent_outputs=act_dim, # Number of actions per agent
            n_agents=self.n_agents, 
            centralised=False, # Policy either centralised or decentralised (True: MADDPG; False: IDDPG)
            share_params=True,
            device=device,
            depth=2, # Number of hidden layers in network
            num_cells=256, # Number of neurons per hidden layer
        )

        # Policy module that takes observations as input and outputs bounded actions
        self.policy = TensorDictModule(
            nn.Sequential(self._actor_core, TanhRescale(low, high)),
            in_keys=[("agents","observation")],
            out_keys=[env.action_key],
        )

        self.policy_target = copy.deepcopy(self.policy) # Creates a target policy network identical to the main policy
        
        # Use Adam optimizer for training actor parameters (weights of the MLP)
        self.act_optimizer = torch.optim.Adam(
            self._actor_core.parameters(), lr=actor_lr
        )

        # Critic core
        critic_core = MultiAgentMLP(
            n_agent_inputs=obs_dim + act_dim, # Input = observation + action
            n_agent_outputs=1, # Value estimate
            n_agents=self.n_agents,
            centralised=True, # Polcy either centralised or decentralised (True: MADDPG; False: IDDPG)
            share_params=True,
            device=device,
            depth=2, # Number of hidden layers in network
            num_cells=256, # Number of neurons per hidden layer
            activation_class=torch.nn.Tanh,
        )

        # Policy module that outputs predicted value estimate to ("agents", "qval")
        self.critic = TensorDictModule(
            critic_core,
            in_keys=[("agents","observation"), ("agents","action")],
            out_keys=[("agents","qval")],
        )

        self.critic_target = copy.deepcopy(self.critic)
        
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

        # Replay buffer (experience replay)
        self.buffer = ReplayBuffer(
            storage=LazyTensorStorage(buffer_size, device=device),
            sampler=SamplerWithoutReplacement(),
            batch_size=batch_size,
        )

    def preprocess_batch(self, data): # DDPG does not use
        pass

    # Main DDPG update
    def update(self, data, minibatch_size=None, num_epochs=1, max_grad_norm=1.0):
        
        # Reshape flattened collected data [batch_size * num_agents, obs_dim] into [batch_size, num_agents, obs_dim]
        obs = data.get(("agents","observation"))
        if obs.ndim == 2:
            Bf, od = obs.shape
            B = Bf // self.n_agents
            td = TensorDict({
                ("agents","observation"):        obs.view(B, self.n_agents, od),
                ("agents","action"):             data.get(("agents","action")).view(B, self.n_agents, -1),
                ("next","agents","observation"): data.get(("next","agents","observation")).view(B, self.n_agents, -1),
                ("next","agents","episode_reward"):      data.get(("next","agents","episode_reward")).view(B, self.n_agents, 1),
                ("next","agents","done"):        data.get(("next","agents","done")).view(B, self.n_agents, 1),
            }, batch_size=[B, self.n_agents])
        else:
            td = data
        
        self.buffer.extend(td) # Add batch to replay buffer

        # Sample minibatch from buffer: obs, act, next_obs, reward, done extracted from sampled batch
        for _ in range(num_epochs):
            batch = self.buffer.sample()
            obs      = batch.get(("agents","observation"))
            act      = batch.get(("agents","action"))
            next_obs = batch.get(("next","agents","observation"))
            rew      = batch.get(("next","agents","episode_reward"))
            done     = batch.get(("next","agents","done")).float()
            B = obs.shape[0]

            # Update critic: minimize MSE between predicted and target q value
            with torch.no_grad():
                next_td = batch.select(*batch.keys())
                next_td = self.policy_target(next_td) # Get target actions from target policy
                next_act = next_td.get(self.env.action_key)

                # Pass next_obs, next_act into critic target to get next q value
                td_in = TensorDict({
                    ("agents","observation"): next_obs,
                    ("agents","action"):      next_act,
                }, batch_size=[B, self.n_agents])
                q_next = self.critic_target(td_in).get(("agents","qval"))
                rew_sum  = rew.sum(dim=1, keepdim=True)
                done_any = done.any(dim=1, keepdim=True).float()
                y = rew_sum + self.gamma * (1 - done_any) * q_next # Compute target

            # Compute current predicted q value using obs, action
            td_curr = TensorDict({
                ("agents","observation"): obs,
                ("agents","action"):      act,
            }, batch_size=[B, self.n_agents])
            q_pred = self.critic(td_curr).get(("agents","qval"))

            # Minimize MSE between predicted and target q values
            loss_c = F.mse_loss(q_pred, y)
            self.critic_optimizer.zero_grad()
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
            self.critic_optimizer.step()

            # Update actor: policy update to output actions that maximize critic's q value
            pi_td = batch.select(*batch.keys())
            pi_td = self.policy(pi_td)
            pred_act = pi_td.get(self.env.action_key) # Get predicted actions from current policy

            # Feed obs and predicted action into critic to get estimated q value
            td_pi = TensorDict({
                ("agents","observation"): obs,
                ("agents","action"):      pred_act,
            }, batch_size=[B, self.n_agents])
            
            # Actor loss = negative q value to perform gradient ascent
            q_pi = self.critic(td_pi).get(("agents","qval"))
            loss_a = -q_pi.mean()

            # Backpropagation update
            self.act_optimizer.zero_grad()
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(self._actor_core.parameters(), max_grad_norm)
            self.act_optimizer.step()

            # Soft target updates to slowly move target networks toward main networks
            for p, tp in zip(self.policy.parameters(), self.policy_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)
