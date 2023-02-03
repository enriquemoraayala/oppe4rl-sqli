from math import frexp
from gym_waf.envs.waf_brain_env import WafBrainEnv
from gym.envs.registration import register
import os
import ray.rllib.agents.ppo as ppo
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
import ray
from ray.tune.registry import register_env
import os
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
# number of steps in an episode
MAXTURNS = 10

# path to the list of sql injection payloads
DATASET = os.path.join(os.path.dirname(__file__), 'gym_waf', 'data',
                       'sqli-1waf.csv')
N_ITER = 30
CHECKPOINT_ROOT = '/Users/ESMoraEn/repositories/rl-waf/checkpoints'

if __name__ == '__main__':
    # initialize ray, for connecting to cluster use ray.init(address="auto")
    # ray.init()

    ray.init(_node_ip_address="0.0.0.0")
    # gym factory

    def env_creator(env_config):
        return WafBrainEnv(DATASET, MAXTURNS)

    register_env("rl-waf-env", env_creator)

    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["num_gpus"] = 0
    config["num_workers"] = 1
    agent = ppo.PPOTrainer(config, env="rl-waf-env")

    s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

    for n in range(N_ITER):
        result = agent.train()
        file_name = agent.save(CHECKPOINT_ROOT)

        print(s.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            file_name
        ))

    # tune.run(PPOTrainer,
    #          config={
    #                  "env": "my_env",
    #                  "num_gpus": 0,
    #                  "num_workers": 1,
    #                  "log_level": "DEBUG",
    #                  # experimented with different models, didn't make
    #                  # much difference
    #                  "model": {"fcnet_hiddens": [512, 512, 256], },
    #                 },
    #          checkpoint_freq=10)
