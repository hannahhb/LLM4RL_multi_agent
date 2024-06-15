import os
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from pettingzoo.mpe import simple_spread_v3

from stable_baselines3.common.callbacks import BaseCallback


from stable_baselines3.common.logger import configure

def make_env():
    # This configures the environment with specific parameters
    # N is the number of agents, max_cycles is the number of steps in each episode
    env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=25)
    return env

# Convert pettingzoo environment to gym environment
env = make_env()
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

# Monitor and normalize the environment
env = VecMonitor(env)
env = VecNormalize(env, norm_obs=True, norm_reward=True)


tmp_path = "./ppo_simple_spread_sb3/"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

n_training_envs=1
log_dir = "./ppo_simple_spread_tensorboard/"


model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
model.set_logger(new_logger)

model.learn(total_timesteps=5000000 )
model.save("ppo_simple_spread")

# evaluate
# env = make_env()
# env = ss.pettingzoo_env_to_vec_env_v1(env)
# env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')
# env = VecMonitor(env)
# env = VecNormalize(env, norm_obs=True, norm_reward=True, training=False)

# model = PPO.load("ppo_simple_spread")

# obs = env.reset()
# for _ in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

