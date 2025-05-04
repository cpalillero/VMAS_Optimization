# PPO.py

# Import packages
import torch
from torch.cuda.amp import autocast, GradScaler
from tensordict.nn import TensorDictModule, NormalParamExtractor, set_composite_lp_aggregate
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

set_composite_lp_aggregate(False).set() # Turn off automatic aggregation of log-probs across actions
# Multi-agent needs per-agent log-probs instead of summed across agents

class PPOAgent: # PPO Agent class
    # Initialize agent with PPO parameters
    def __init__(
        self,
        env,
        device,
        actor_lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 1e-4,
        gamma: float = 0.99,
        lmbda: float = 0.9,
        minibatch_size: int = 400,
        num_epochs: int = 4,
    ):
        # Set up device, environment, number of agents
        self.device = device
        self.env = env
        self.n_agents = env.n_agents
        
        # Set up observation and action dimensions to size networks
        self.obs_dim = env.observation_spec["agents", "observation"].shape[-1]
        self.act_dim = env.action_spec.shape[-1]

        # Actor network
        self.policy_net = torch.nn.Sequential( # Actor is a shared MLP that takes observations and outputs (loc,scale) for Gaussian distribution
            MultiAgentMLP(
                n_agent_inputs=self.obs_dim, # Number of observations per agent
                n_agent_outputs=2 * self.act_dim, # Mean and standard deviation
                n_agents=self.n_agents,
                centralised=False, # Policy either centralised or decentralised (True: MAPPO; False: IPPO)
                share_params=True, # Parameters shared across all agents
                device=self.device,
                depth=2, # Number of hidden layers in network
                num_cells=256, # Number of neurons per hidden layer
                activation_class=torch.nn.Tanh,
            ),
            NormalParamExtractor(), # Splits list layer output into mean/standard deviation
        )

        # Converts to probabilistic policy, returns tanh(Normal(loc,scale)) bounded by action space
        self.policy_module = TensorDictModule( # Read inputs from agents, observations and produce outputs (agents, loc) and (agents, scale)
            self.policy_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )
        self.policy = ProbabilisticActor( # Takes means and scales and builds TanhNormal distribution, then samples actions from distribution
            module=self.policy_module,
            spec=env.action_spec_unbatched,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[env.action_key],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": env.full_action_spec_unbatched[env.action_key].space.low,
                "high": env.full_action_spec_unbatched[env.action_key].space.high,
            },
            return_log_prob=True, # Outputs log-prob for PPO loss
        )

        # Critic network
        critic_mlp = MultiAgentMLP( # Critic is a shared MLP that outputs a scaler state value per agent
            n_agent_inputs=self.obs_dim, # Number of observations per agent
            n_agent_outputs=1, # Value estimate for agent's observation
            n_agents=self.n_agents,
            centralised=True, # Policy either centralised or decentralised (True: MAPPO; False: IPPO)
            share_params=True, # Parameters shared across all agents
            device=self.device,
            depth=2, # Number of hidden layers in network
            num_cells=256, # Number of neurons per hidden layer
            activation_class=torch.nn.Tanh,
        )
        self.critic = TensorDictModule(
            critic_mlp,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "state_value")],
        )

        # PPO loss
        self.loss_module = ClipPPOLoss( # Uses clipped surrogate objective, adds emtropy bonus, and includes critic loss
            actor_network=self.policy,
            critic_network=self.critic,
            clip_epsilon=clip_epsilon,
            entropy_coef=entropy_coef,
            normalize_advantage=False,
        )
        self.loss_module.set_keys( # Keys for accessing tensorsin TensorDict
            reward=env.reward_key,
            action=env.action_key,
            value=("agents", "state_value"),
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )
        self.loss_module.make_value_estimator( # Initialize Generalized Advantage Estimation (GAE) to compute advantage function
            ValueEstimators.GAE,
            gamma=gamma,
            lmbda=lmbda,
        )

        # Use Adam optimizer and mixed precision (AMP)
        self.optimizer = torch.optim.Adam(
            self.loss_module.parameters(), lr=actor_lr
        )
        self.scaler = GradScaler()

        # Replay buffer (PPO on-policy but used to sample mini-batches across multiple epochs)
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(30000, device=self.device),
            sampler=SamplerWithoutReplacement(),
            batch_size=minibatch_size,
        )

        self._num_epochs = num_epochs
        self._minibatch_size = minibatch_size

    def preprocess_batch(self, data): # Compute advantage function before training 
        with torch.no_grad():
            self.loss_module.value_estimator( # Update advantage and value_target
                data,
                params=self.loss_module.critic_network_params,
                target_params=self.loss_module.target_critic_network_params,
            )

    # Main PPO update
    def update(self, data, minibatch_size=None, num_epochs=None, max_grad_norm=1.0):
        mb_size = minibatch_size or self._minibatch_size
        n_epochs = num_epochs or self._num_epochs

        self.replay_buffer.extend(data) # Extend replay buffer with new data
        # For each epoch : For each minibatch : Compute loss_objective + loss_critic + loss_entropy
        # Then use mixed precision backwards
        # Then clip gradients
        # Then optimizer step
        for _ in range(n_epochs):
            for _ in range(len(self.replay_buffer) // mb_size):
                subdata = self.replay_buffer.sample()
                with autocast():
                    losses = self.loss_module(subdata)
                    loss = (
                        losses["loss_objective"]
                        + losses["loss_critic"]
                        + losses["loss_entropy"]
                    )
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.loss_module.parameters(), max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        self.replay_buffer.empty()  # Empty replay buffer for next iteration
