
import numpy as np

from openrl.configs.config import create_config_parser
from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent
from openrl.envs.wrappers import GIFWrapper # used to generate gif


def train():
    # create environment
    env_num = 100
    env = make(
        "simple_spread",
        env_num=env_num,
        asynchronous=True,
        num_agents = 6
    )
    # create the neural network
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()
    net = Net(env, cfg=cfg)
    # initialize the trainer
    agent = Agent(net, use_wandb=True)
    # start training, set total number of training steps to 5000000
    agent.train(total_time_steps=5000000)
    env.close()
    agent.save("./ppo_agent/")
    return agent

def test():
    # Create MPE environment.
    env = make("simple_spread", env_num=4)
    # Use GIFWrapper to generate gifs.
    env = GIFWrapper(env, "test_simple_spread.gif")
    agent = Agent(Net(env))  # Create an intelligent agent.
    # Load the trained model.
    agent.load('./ppo_agent/')
    # Begin to test.
    obs, _ = env.reset()
    while True:
        action, _ = agent.act(obs)
        obs, r, done, info = env.step(action)
        if done.any():
            break
    env.close()

if __name__ == "__main__":
    agent = train()
    # test()