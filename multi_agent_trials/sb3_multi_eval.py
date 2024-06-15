import pandas as pd
import numpy as np

# Load the CSV data
data = pd.read_csv('ppo_simple_spread_sb3/progress.csv')  # Update with the actual path to your CSV file

# Extract relevant columns
rewards = data['rollout/ep_rew_mean']
timesteps = data['time/total_timesteps']

mean_reward = rewards.mean()
std_reward = rewards.std()
normalized_rewards = (rewards - mean_reward) / std_reward

# Calculate AULC using the trapezoidal rule on the normalized rewards
aulc = np.trapz(normalized_rewards, timesteps)

print(f"Area Under the Learning Curve (AULC): {aulc}")
