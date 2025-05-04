# optimize_sac.py

# Import packages
import optuna
import torch
from Agents.SAC import SACAgent
from train import train

# Fixed environment settings
scenario        = "reverse_transport" # VMAS scenario
n_agents        = 5 # Number of agents in scenario
max_steps       = 500 # Maximum timesteps per environment
frames_per_batch= 128_000 # Total number of timesteps across all parallel environments before updating policy
n_iters         = 25 # Total training iterations/policy updates
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    # Sample hyperparameters
    actor_lr     = trial.suggest_loguniform("actor_lr",    1e-5, 1e-2) # Actor learning rate
    critic_lr    = trial.suggest_loguniform("critic_lr",   1e-5, 1e-2) # Critic learning rate
    gamma        = trial.suggest_uniform("gamma",         0.90, 0.999) # Discount factor
    tau          = trial.suggest_uniform("tau",           1e-3, 1e-1) # Soft target update rate
    batch_size   = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
    buffer_size  = trial.suggest_categorical("buffer_size",[10000, 50000, 100000])
    num_epochs   = trial.suggest_int("num_epochs",        1, 10)

    print(
        f"[Trial {trial.number}] "
        f"actor_lr={actor_lr:.4g}, critic_lr={critic_lr:.4g}, "
        f"γ={gamma:.3f}, τ={tau:.3f}, "
        f"buf={buffer_size}, bs={batch_size}, ep={num_epochs}"
    )

    # Agent constructor
    def make_agent(env, device):
        return SACAgent(
            env, device,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            tau=tau,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )
    
    # Train the agent using train.py
    try:
        rewards = train(
            make_agent,
            scenario,
            n_agents,
            max_steps,
            frames_per_batch,
            n_iters=25,
            device=device, num_epochs=num_epochs,
        )
        return sum(rewards[-5:]) / 5 # Return average reward of last five iterations
    except AssertionError as e: # If error occurs during optimization sweep
        trial.set_user_attr("error", str(e))
        return -1e6 # Return very low reward 

# Run optimization sweep
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize") # Create Optuna study with goal of maximizing returned score
    study.optimize(objective, n_trials=25) # Run for 25 trials
    df = study.trials_dataframe(attrs=("number","value","params")) # Save trial number, corresponding value, and corresponding hyperparameter values
    df.to_csv("sac_results.csv", index=False) # Save results to csv file
    print("Best params:", study.best_params)
    print("Best score:", study.best_value)