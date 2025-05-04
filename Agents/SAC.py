# SAC.py

# Import packages
import copy
import torch
import torch.nn.functional as F
from torch import nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, NormalParamExtractor
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement


class SACAgent: # SAC Agent class
    def __init__(
        self,
        env,
        device,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99, # Discount factor
        tau: float = 0.005, # Soft target update rate (how quickly networks update towards learned networks)
        alpha_init: float = 0.01, # Initial entropy weight
        buffer_size: int = 100_000,
        batch_size: int = 1024,
    ):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.n_agents = env.n_agents
        self.alpha = alpha_init


        # env.full_action_spec_unbatched: Dictionary mapping keys (agents, action, etc.) to action space specification
        # env.action_key: Action space specification
        # Rescaling actor's output to be within [low, high] bounds; regardless of environment being used
        low = env.full_action_spec_unbatched[env.action_key].space.low
        high = env.full_action_spec_unbatched[env.action_key].space.high
        
        # Set up observation and action dimensions to size networks
        obs_dim = env.observation_spec["agents", "observation"].shape[-1]
        act_dim = env.action_spec.shape[-1]

        # Actor network
        actor_net = torch.nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=obs_dim, # Number of observations per agent
                n_agent_outputs=2 * act_dim, # Mean and standard deviation
                n_agents=self.n_agents,
                centralised=False, # Policy either centralised or decentralised (True: MASAC; False: ISAC)
                share_params=True,
                device=device,
                depth=2, # Number of hidden layers in network
                num_cells=256, # Number of neurons per hidden layer
                activation_class=torch.nn.Tanh,
            ),
            NormalParamExtractor(), # Splits loc into [mean_x, mean_y] and scale into [log_std_x, log_std_y]
        )

        # Converts to probablistic policy, returns (tanh(Normal(loc,scale)) bounded by action space
        self.actor_module = TensorDictModule( # Read inputs from agents, observations and produce outputs (agents, loc) and (agents, scale)
            actor_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )
        self.policy = ProbabilisticActor( # Takes means and scales and builds TanhNormal distribution, then samples actions from distribution
            
            module=self.actor_module,
            spec=env.action_spec_unbatched,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[env.action_key],
            distribution_class=TanhNormal,
            distribution_kwargs={"low": low, "high": high},
            return_log_prob=True,
        )

        # Critic networks
        critic_input_dim = obs_dim + act_dim # Critic input = concatenation of observation and action
        self.critic1 = TensorDictModule( # First critic network
            MultiAgentMLP(
                n_agent_inputs=critic_input_dim, # Number of observations per agent
                n_agent_outputs=1, # Value estimate for agent's observation
                n_agents=self.n_agents,
                centralised=True, # Policy either centralised or decentralised (True: MASAC; False: ISAC)
                share_params=True,
                device=device,
                depth=2, # Number of hidden layers in network
                num_cells=256, # Number of neurons per hidden layer
                activation_class=torch.nn.Tanh,
            ),
            in_keys=[("agents", "observation"), ("agents", "action")],
            out_keys=[("agents", "q1")],
        )
        self.critic2 = TensorDictModule( # Second critic network
            MultiAgentMLP(
                n_agent_inputs=critic_input_dim, # Number of observations per agent
                n_agent_outputs=1, # Value estimate for agent's observation
                n_agents=self.n_agents,
                centralised=True, # Policy either centralised or decentralised (True: MASAC; False: ISAC)
                share_params=True,
                device=device,
                depth=2, # Number of hidden layers in network
                num_cells=256, # Number of neurons per hidden layer
                activation_class=torch.nn.Tanh,
            ),
            in_keys=[("agents", "observation"), ("agents", "action")],
            out_keys=[("agents", "q2")],
        )

        # Target critic networks
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        # Actor and critic optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor_module.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # Entropy coefficient
        self.alpha = alpha_init

        # Replay buffer (experience replay)
        self.buffer = ReplayBuffer(
            storage=LazyTensorStorage(buffer_size, device=device),
            sampler=SamplerWithoutReplacement(),
            batch_size=batch_size,
        )

    def preprocess_batch(self, data): # SAC does not use
        pass 

    # Main SAC update
    def update(self, data, minibatch_size=None, num_epochs=1, max_grad_norm=1.0):
        # Reshape flattened collected data [batch_size * num_agents, obs_dim] into [batch_size, num_agents, obs_dim]
        obs = data.get(("agents", "observation"))
        if obs.ndim == 2:
            Bf, od = obs.shape
            B = Bf // self.n_agents
            td = TensorDict({
                ("agents", "observation"): obs.view(B, self.n_agents, od),
                ("agents", "action"): data.get(("agents", "action")).view(B, self.n_agents, -1),
                ("next", "agents", "observation"): data.get(("next", "agents", "observation")).view(B, self.n_agents, -1),
                ("next", "agents", "episode_reward"): data.get(("next", "agents", "episode_reward")).view(B, self.n_agents, 1),
                ("next", "agents", "done"): data.get(("next", "agents", "done")).view(B, self.n_agents, 1),
            }, batch_size=[B, self.n_agents])
        else:
            td = data
 
        self.buffer.extend(td) # Add batch to replay buffer

        # Sample minibatch from buffer: obs, act, next_obs, reward, done extracted from sampled batch
        for _ in range(num_epochs):
            batch = self.buffer.sample()
            obs = batch.get(("agents", "observation"))
            act = batch.get(("agents", "action"))
            next_obs = batch.get(("next", "agents", "observation"))
            rew = batch.get(("next", "agents", "episode_reward"))
            done = batch.get(("next", "agents", "done")).float()
            B = obs.shape[0]

            # Next-state action sampling (policy forward pass)
            pi_td = TensorDict({("agents", "observation"): next_obs}, batch_size=[B, self.n_agents])
            pi_td = self.policy(pi_td)
            next_act = pi_td.get(self.env.action_key)
            logp_next = pi_td.get("sample_log_prob")
            logp_next = logp_next.view(B, self.n_agents, -1)

            # Compute target q values
            td1 = TensorDict({
                ("agents", "observation"): next_obs,
                ("agents", "action"): next_act,
            }, batch_size=[B, self.n_agents])
            q1_target = self.critic1_target(td1).get(("agents", "q1"))
            q2_target = self.critic2_target(td1).get(("agents", "q2"))
            min_q = torch.min(q1_target, q2_target) # SAC double-q learning method: take minimum q-value to reduce overestimation bias

            # Target q value (Bellman backup)
            alpha = self.log_alpha.exp() if self.automatic_entropy_tuning else self.alpha
            target_q = min_q - alpha * logp_next
            rew_sum = rew.sum(dim=1, keepdim=True)
            done_any = done.any(dim=1, keepdim=True).float()
            y = rew_sum + self.gamma * (1 - done_any) * target_q # Bellman backup (Target for critic update)


            # Critic update
            curr_td1 = TensorDict({
                ("agents", "observation"): obs,
                ("agents", "action"): act,
            }, batch_size=[B, self.n_agents])
            # Evalute current critics (obs, act)
            q1_pred = self.critic1(curr_td1).get(("agents", "q1"))
            q2_pred = self.critic2(curr_td1).get(("agents", "q2"))

            critic1_loss = F.mse_loss(q1_pred, y)
            critic2_loss = F.mse_loss(q2_pred, y)
            loss_c = critic1_loss + critic2_loss # Compute MSE between predicted q and target y
            # Backpropagate and optimizer for both critics
            self.critic1_optimizer.zero_grad()
            self.critic2_optimizer.zero_grad()
            loss_c.backward() 
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_grad_norm)
            self.critic1_optimizer.step()
            self.critic2_optimizer.step()


            # Actor update
            # Sample current policy actions at obs
            pi_td = TensorDict({("agents", "observation"): obs}, batch_size=[B, self.n_agents])
            pi_td = self.policy(pi_td)
            new_act = pi_td.get(self.env.action_key)
            logp = pi_td.get("sample_log_prob")
            logp = logp.view(B, self.n_agents, -1)
            td_pi = TensorDict({
                ("agents", "observation"): obs,
                ("agents", "action"): new_act,
            }, batch_size=[B, self.n_agents])
            # Predicted q-values at sampled action
            q1_pi = self.critic1(td_pi).get(("agents", "q1"))
            q2_pi = self.critic2(td_pi).get(("agents", "q2"))
            min_q_pi = torch.min(q1_pi, q2_pi)

            actor_loss = (alpha * logp - min_q_pi).mean() # Actor loss

            # Backpropagation update
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_grad_norm)
            self.actor_optimizer.step()

            # Soft target updates to slowly move target networks toward main networks
            for p, tp in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)
            for p, tp in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)
