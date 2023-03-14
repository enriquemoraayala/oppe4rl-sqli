import argparse
import json
import os
import pandas as pd
from azureml.core import Run

import ray.rllib.agents.ppo as ppo
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
#from ray.rllib.algorithms.dqn.dqn import DQNTrainer
import ray
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
#import tensorflow as tf
import tensorflow as tf
import numpy as np

import torch
from gym_waf.envs.waf_brain_env import WafBrainEnv
from ray.rllib.algorithms import Algorithm
from ray.rllib.models.preprocessors import get_preprocessor


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


def generate_df():
    df_results = pd.DataFrame(columns=[
                              'episode', 'step', 'original_payload', 'state', 'action',
                              'next_state', 'reward', 'win'])
    df_results_emb = pd.DataFrame(columns=[
                                  'episode', 'step', 'state_emb', 'next_state_emb'
                                ])
    df_results = df_results.astype({
        "episode": int, "step": int, "original_payload": "object", "action": int, "state": "object", "next_state": "object",
        "reward": float, "win": int})
    df_results_emb = df_results_emb.astype({
        "episode": int, "step": int, "state_emb": float, "next_state_emb": float
    })
    return df_results, df_results_emb

def main(args):
    tf.compat.v1.enable_eager_execution()
    #tf.compat.v1.disable_v2_behavior()
    ray.init(local_mode=args.local_mode)
    run = Run.get_context(allow_offline=True)

    env_config_fpath = args.env_config
    # Read configurations
    with open(env_config_fpath, "rb") as f:
        env_config = json.load(f)

    payloads_fpath = os.path.join(os.path.dirname(
        __file__), 'gym_waf', 'data', env_config["payloads"])
    env_config["payloadsPath"] = payloads_fpath
    # select 1 sql random
    # run actions...
    register_env("rl-waf", env_creator)

    env = env_creator(env_config)
    prep = get_preprocessor(env.observation_space)(env.observation_space)
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.shape
    
 
    algo = Algorithm.from_checkpoint("./ckpt_ppo_agent_tf2/checkpoint_000006")
    
    #generating some trajectories and evaluating the trained policy
    num_trajectories = 200
    
    df_results, df_results_emb = generate_df()
    # ppo_policy = algo.get_policy()

    for episode in range(num_trajectories):
        done = False
        episode_reward = 0
        obs = env.reset()
        step = 0
        while not done:
            action_direct = algo.compute_single_action(obs)
            # logits, _ = ppo_policy.model({"obs": tf.expand_dims(tf.convert_to_tensor(obs), axis=0)})
            # dist = ppo_policy.dist_class(logits, ppo_policy.model)
            obs, reward, done, env_info = env.step(action_direct)
            df_results = pd.concat(
                [df_results,
                 pd.DataFrame([[episode, step, env.orig_payload, env_info['previous_payload'],
                                action_direct, env_info["payload"],
                                reward, env_info['win']]],
                                columns=df_results.columns)],
                axis=0,
                join="inner",
                ignore_index=True)
            df_results_emb = pd.concat(
                [df_results_emb,
                 pd.DataFrame([[episode, step, env_info['emb_payload'], env_info["emb_prev_payload"]]],
                                columns=df_results_emb.columns)],
                axis=0,
                join="inner",
                ignore_index=True)
            episode_reward += reward
            step += 1

        # print('Final payload %s' %env_info['payload'])
        s = "episode {:d} reward {:6.2f} len {:6.2f}"
        print(s.format(
                episode,
                episode_reward,
                step
            ))
    csv_path = 'outputs_ppo/run_history_rllib.csv'
    df_results.to_csv(csv_path, sep=';')
    csv_path = 'outputs_ppo/run_history_rllib_emb.csv'
    df_results_emb.to_csv(csv_path, sep=';')
    csv_path = 'outputs_ppo/run_history_rllib_emb_idx.csv'
    df_results_emb[['episode', 'step']].to_csv(csv_path, sep=';')
    state_emb = np.array(df_results_emb['state_emb'])
    csv_path = 'outputs_ppo/emb_states_rllib.npy'
    np.save(csv_path, state_emb)
    print('Files saved! Process done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train RL agent on WAF-Brain Environment with RLLib")
    parser.add_argument("--env-config", type=str, help="Path to configuration file of the envionment.",
                        default="/home/azureuser/cloudfiles/code/Users/Enrique.Mora/rl4pentest/src/config/env.json")
    parser.add_argument("--local-mode", action="store_true", help="Init Ray in local mode for easier debugging.")
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    main(args)