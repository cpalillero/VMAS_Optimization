Project structure:

 > path/to/project/

  > Agents/            # Contains PPO, DDPG, and SAC agents
  
  > main.py            # Run single agent with fixed parameters
  
  > train.py           # Training logic
  
  > make_env.py        # Environment setup
  
  > optimize_ppo.py    # Optimize PPO using Optuna
  
  > optimize_ddpg.py   # Optimize DDPG using Optuna
  
  > optimize_sac.py    # Optimize SAC using Optuna

Usage instructions:
1. Confirm if centralised or decentralised (MA vs. I) agent is desired
 > Must be changed in respective script (PPO.py, DDPG.py, SAC.py)
2. If running for a single trial, execute main.py
3. If optimizing agent across multiple trials, execute respective optimization script

Dependencies:
torch
torchrl
tensordict
optuna
vmas
matplotlib
tqdm
pandas
