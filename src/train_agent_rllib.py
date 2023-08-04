import argparse
import json
import os
from azureml.core import Run

import ray.rllib.agents.ppo as ppo
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
#from ray.rllib.algorithms.dqn.dqn import DQNTrainer
import ray
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
import tensorflow as tf
from gym_waf.envs.waf_brain_env import WafBrainEnv

#tf.compat.v1.disable_eager_execution()
# number of steps in an episode

def main(args):
    # tf.compat.v1.enable_eager_execution()
    ray.init(local_mode=args.local_mode)
    run = Run.get_context(allow_offline=True)

    env_config_fpath = args.env_config
    #agent_config_fpath = args.agent_config
    # Read configurations
    with open(env_config_fpath, "rb") as f:
        env_config = json.load(f)
    #with open(agent_config_fpath, "rb") as f:
    #    agent_config = json.load(f)

    payloads_fpath = os.path.join(os.path.dirname(
        __file__), 'gym_waf', 'data', env_config["payloads"])
    env_config["payloadsPath"] = payloads_fpath

    def env_creator(env_config: dict):
        max_steps = env_config["steps"]
        payloads_fpath = env_config["payloadsPath"]
        allow_payload_actions = env_config["allowPayloadActions"]
        feature_extractor = env_config["featureExtractor"]
        reward_name = env_config["reward"]
        reward_win_val = env_config["rewardWin"]
        env = WafBrainEnv(payloads_file=payloads_fpath, maxturns=max_steps, feature_extractor=feature_extractor,
                          payload_actions=allow_payload_actions, reward_name=reward_name, reward_win_val=reward_win_val)
        return env

    register_env("rl-waf", env_creator)

    env = env_creator(env_config)
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.shape
    run.tag("actionSpaceSize", action_space_size)
    run.tag("envStateSpaceSize", state_space_size)
    for k, v in env_config.items():
        run.tag(k, v)

    scores = []
    agent = "PPO"
    run.tag("agent", agent)

    config = {
        "framework": "torch",
        "env": "rl-waf",
        "env_config": env_config,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 0,
        "log_level": "WARN",
        "horizon": 30
    }

    trainer = PPOTrainer(config)
    
    lengths = []
    # for n in range(int(500 / 32)):
    for n in range(10):
        result = trainer.train()
        if n%5 == 0:
            print(pretty_print(result))
            file_name = trainer.save('ckpt_ppo_agent_torch')
        s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"
        lengths += result["hist_stats"]["episode_lengths"]
        print(s.format(
            n + 1,
            result["episode_reward_min"],
            result["episode_reward_mean"],
            result["episode_reward_max"],
            result["episode_len_mean"],
            file_name
        ))
    print(0)
    print(pretty_print(result))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train RL agent on WAF-Brain Environment with RLLib")
    parser.add_argument("--env-config", type=str, help="Path to configuration file of the envionment.",
                        default="/home/azureuser/cloudfiles/code/Users/Enrique.Mora/rl4pentest/src/config/env.json")
    parser.add_argument("--local-mode", action="store_true", help="Init Ray in local mode for easier debugging.")
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    main(args)
