# optimize_ppo.py

# Import packages
import optuna
import torch
from train import train
from Agents.PPO import PPOAgent

# Fixed environment settings
scenario        = "reverse_transport" # VMAS scenario
n_agents        = 5 # Number of agents in scenario
max_steps       = 500 # Maximum timesteps per environment
frames_per_batch= 128_000 # Total number of timesteps across all parallel environments before updating policy
n_iters         = 25 # Total training iterations/policy updates
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimization function called by Optuna (returns scalar metric that will be optimized)
def objective(trial):
    # Sample hyperparameters
    actor_lr      = trial.suggest_loguniform("actor_lr",      1e-5,    1e-3) # Actor learning rate
    entropy_coef  = trial.suggest_loguniform("entropy_coef",  1e-4,    1e-2) # Entropy coefficient
    clip_epsilon  = trial.suggest_uniform("clip_epsilon",   0.1,    0.3) # PPO clip threshold
    gamma         = trial.suggest_uniform("gamma",           0.95,    0.999) # Discount factor
    lmbda         = trial.suggest_uniform("lmbda",           0.8,     0.99) # GAE lambda
    minibatch_size= trial.suggest_categorical("minibatch_size", [256,512,1024,2048])
    num_epochs    = trial.suggest_int("num_epochs",           1,      10)

    print(f"[Trial {trial.number}] "
          f"lr={actor_lr:.4g}, clip={clip_epsilon:.3f}, ent={entropy_coef:.4g}, "
          f"γ={gamma:.3f}, λ={lmbda:.3f}, mb={minibatch_size}, ep={num_epochs}")

    # Agent constructor
    def make_agent(env, device):
        return PPOAgent(env, device,
                        actor_lr=actor_lr,
                        clip_epsilon=clip_epsilon,
                        entropy_coef=entropy_coef,
                        gamma=gamma,
                        lmbda=lmbda,
                        minibatch_size=minibatch_size,
                        num_epochs=num_epochs)
    
    # Train the agent using train.py
    try:
        rewards = train(
            make_agent, scenario, n_agents, max_steps,
                        frames_per_batch, n_iters, device, minibatch_size=minibatch_size,
            num_epochs=num_epochs)
        return sum(rewards[-5:]) / 5 # Return average reward of last five iterations
    
    except AssertionError as e: # If error occurs during optimization sweep
        trial.set_user_attr("error", str(e))
        return -1e6 # Return very low reward 

# Run optimization sweep
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize") # Create Optuna study with goal of maximizing returned score
    study.optimize(objective, n_trials=25)   # Run for 25 trials
    df = study.trials_dataframe(attrs=("number","value","params")) # Save trial number, corresponding value, and corresponding hyperparameter values
    df.to_csv("ppo_results.csv",index=False) # save results to csv file
    print("Best params:", study.best_params)
    print("Best score: ", study.best_value)
