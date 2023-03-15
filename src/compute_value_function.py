import argparse
import json
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
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

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hid1 = torch.nn.Linear(794, 256)
        self.hid2 = torch.nn.Linear(256, 128)
        self.hid3 = torch.nn.Linear(128, 32)
        self.oupt = torch.nn.Linear(32, 1)

        torch.nn.init.xavier_uniform_(self.hid1.weight)
        torch.nn.init.zeros_(self.hid1.bias)
        torch.nn.init.xavier_uniform_(self.hid2.weight)
        torch.nn.init.zeros_(self.hid2.bias)
        torch.nn.init.xavier_uniform_(self.hid3.weight)
        torch.nn.init.zeros_(self.hid3.bias)
        torch.nn.init.xavier_uniform_(self.oupt.weight)
        torch.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = torch.relu(self.hid1(x))
        z = torch.relu(self.hid2(z))
        z = torch.relu(self.hid3(z))
        z = self.oupt(z)  # no activation
        return z

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

def load_df(path='./outputs_ppo', agent_type='ppo'):
    if agent_type == 'ppo':
        df = pd.read_csv(path + '/run_history_rllib.csv', sep=';', index_col=0)
        s2 = path + '/emb_states_rllib.npy'
        np_emb_state = np.load(s2, allow_pickle=True)
    else:
        #reading the first file
        df = pd.read_csv('./outputs_random/run_history_199.csv', sep=';', index_col=0)
        df_emb = pd.read_csv('./outputs_random/run_history_emb_idx_199.csv', sep=';', index_col=0)
        np_emb_state = np.load('./outputs_random/emb_states_199.npy', allow_pickle=True)
        np_emb_next_state = np.load('./outputs_random/next_states_199.npy', allow_pickle=True)

        skeeped_files = []
        #adding all the others
        list1 = [399]
        #list1 = list(range(399, 399, 1))
        for i in list1:
            s1 = './outputs_random/run_history_' + str(i) + '.csv'
            df_ = pd.read_csv(s1, sep=';', index_col=0)
            s2 = './outputs_random/emb_states_' + str(i) + '.npy'
            np_emb_state_ = np.load(s2, allow_pickle=True)
            if np_emb_state_.shape[0] != df_.shape[0]:
                print ('skeeping file %d' %i)
                skeeped_files.append(s1)
                skeeped_files.append(s2)
            else:
                df = pd.concat([df,df_], axis=0, ignore_index=True)
                np_emb_state = np.concatenate((np_emb_state, np_emb_state_))

    return df, np_emb_state



def main(args):
    tf.compat.v1.enable_eager_execution()
    ray.init(local_mode=args.local_mode)

    env_config_fpath = args.env_config
    # Read configurations
    with open(env_config_fpath, "rb") as f:
        env_config = json.load(f)

    payloads_fpath = os.path.join(os.path.dirname(
        __file__), 'gym_waf', 'data', env_config["payloads"])
    env_config["payloadsPath"] = payloads_fpath

    register_env("rl-waf", env_creator)

    env = env_creator(env_config)
    print('Number of actions: %d' %env.action_space.n)
    print('State space dimension: %s' %env.observation_space.shape)
    if args.agent_type == 'ppo':
        algo = Algorithm.from_checkpoint("./ckpt_ppo_agent_tf2/checkpoint_000006")
        print('Checkpoint loaded')
        path_to_trajectories = './outputs_ppo'
        ppo_policy = algo.get_policy()
    else:
        path_to_trajectories = './outputs_random'
    
    #generating some trajectories and evaluating the trained policy
    df, np_emb_state_ = load_df(path_to_trajectories, args.agent_type)
    print(np_emb_state_.shape[0])
    print(df.shape[0])
    assert np_emb_state_.shape[0] == df.shape[0], 'number of steps in trajectories does not match'

    episodes = list(df.episode.unique())
    print('Number of episodes loaded: %d' %len(episodes))

    # calculating expected reward for episode

    
    print('Calculating cum_reward of every episode')
    discount = 0.99 #based in the PPO original paper, default discount

    for ep in episodes:
        df_ = df[df['episode'] == ep].copy()
        df_.sort_values(by=['step'], ascending=False, inplace=True)
        cum_reward = 0.0
        j = 0
        for i,step in df_.iterrows():
            if j == 0:
                cum_reward = step.reward
            else:
                cum_reward = cum_reward + discount * step.reward
            df.at[i,'exp_reward'] = cum_reward
            j += 1
    
    print(df.head())
    print('Calculating Real Value Function')

    J_eps = 0.0
    df_ = df.groupby('episode').first()
    J_eps = df_['exp_reward'].mean()
    print('Real Value function of %s Policy: %.8f' % (args.agent_type, J_eps))

    #loading q_regressor
    print('Loading regressor')
    model = SimpleNet()
    model.load_state_dict(torch.load('./q_regressor_chkp/q_regressor.pth'))
    model.eval()
    print('Q_Regressor loaded')

    print('Computing DM OPPE')
    actions = range(0,26)
    a_t = torch.Tensor(actions).to(torch.int64)
    actions_one_hot = torch.nn.functional.one_hot(a_t)

    values = np.ndarray()
    for ep in episodes:
        df_ = df[df['episode'] == ep].copy()

        probs = np.ndarray()
        q_estimations = np.ndarray()

        for i,step in df_.iterrows():
            obs = np_emb_state_[i]
            action = step['action']
            t_x = torch.cat([obs, actions_one_hot[action]], dim=1)
            x = torch.nn.functional.normalize(x)
            output = model(x)
            q_estimations.append(output)

            logits, _ = ppo_policy.model({"obs": tf.expand_dims(tf.convert_to_tensor(obs), axis=0)})
            probs_ = tf.nn.softmax(logits)
            prob_ = probs_[0][action]
            probs.append(prob_.numpy())
        values.append(np.dot(probs, q_estimations.T))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train RL agent on WAF-Brain Environment with RLLib")
    parser.add_argument("--env-config", type=str, help="Path to configuration file of the envionment.",
                        default="/home/azureuser/cloudfiles/code/Users/Enrique.Mora/rl4pentest/src/config/env.json")
    parser.add_argument("--local-mode", action="store_true", help="Init Ray in local mode for easier debugging.")
    parser.add_argument("--agent_type", default="ppo")
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    main(args)