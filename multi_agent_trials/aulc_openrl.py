import re
import numpy as np

def parse_log_file(file_path):
    episode_rewards = []
    timesteps = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if "rollout_episode_reward" in line:
                match = re.search(r"rollout_episode_reward: ([\-\d\.]+)", line)
                print(match)
                if match:
                    reward = float(match.group(1))
                    episode_rewards.append(reward)
            # elif "Episode:" in line:
            #     match = re.search(r"Episode: (\d+)/", line)
            #     print(match)
            #     if match:
                # episode = int(match.group(1))
                timestep = episode * 2500  # Each episode corresponds to 2500 timesteps
                timesteps.append(timestep)
    
    return timesteps, episode_rewards

def calculate_aulc(timesteps, rewards):
    # Normalize rewards using Z-score normalization
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    normalized_rewards = (rewards - mean_reward) / std_reward

    # Calculate the Area Under the Learning Curve using the trapezoidal rule
    aulc = np.trapz(normalized_rewards, timesteps)
    return aulc

# Path to your log file
log_file_path = 'exp_results/PPOAgent/simple_spread/ppo/run1/log.txt'  # Update this path

# Parse the log file to extract timesteps and rewards
timesteps, rewards = parse_log_file(log_file_path)

# Calculate AULC
aulc = calculate_aulc(timesteps, rewards)
print(f"Normalized Area Under the Learning Curve (AULC): {aulc}")

# # Optional: Save the steps and values to a CSV for further analysis or visualization
# import pandas as pd

# df = pd.DataFrame({'timesteps': timesteps, 'rewards': rewards})
# df.to_csv('reward_curve.csv', index=False)
