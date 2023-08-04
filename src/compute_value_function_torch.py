import argparse
import json
import os
import pandas as pd
import ray
import numpy as np
import torch
import tensorflow as tf

from ray.tune.registry import register_env
from gym_waf.envs.waf_brain_env import WafBrainEnv
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo import PPO


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
    # parece que rllib al guardar el checkpoint guarda tambien los parametros
    # del env y lo vuelve a crear
    # la version 2.2 no permite enviarle el entorno (parece)
    # workaround: voy a forzar el payloads_fpath
    max_steps = env_config["steps"]
    payloads_fpath = env_config["payloadsPath"]
    # payloads_fpath = "/Users/ESMoraEn/repositories/dcg-oppe/src/gym_waf/data/sqli-1k.csv"
    allow_payload_actions = env_config["allowPayloadActions"]
    feature_extractor = env_config["featureExtractor"]
    reward_name = env_config["reward"]
    reward_win_val = env_config["rewardWin"]
    env = WafBrainEnv(payloads_file=payloads_fpath, maxturns=max_steps,
                      feature_extractor=feature_extractor,
                      payload_actions=allow_payload_actions,
                      reward_name=reward_name, reward_win_val=reward_win_val)
    return env


def generate_df():
    df_results = pd.DataFrame(columns=[
                              'episode', 'step', 'original_payload', 'state',
                              'action',
                              'next_state', 'reward', 'win'])
    df_results_emb = pd.DataFrame(columns=['episode', 'step',
                                           'state_emb', 'next_state_emb'
                                           ])
    df_results = df_results.astype({
        "episode": int, "step": int, "original_payload": "object",
        "action": int, "state": "object", "next_state": "object",
        "reward": float, "win": int})
    df_results_emb = df_results_emb.astype({
        "episode": int, "step": int, "state_emb": float,
        "next_state_emb": float
    })
    return df_results, df_results_emb


def load_df(path='./outputs_ppo', agent_type='ppo'):

    df = pd.read_csv(path + '/run_history_rllib.csv', sep=';', index_col=0)
    s2 = path + '/emb_states_rllib.npy'
    np_emb_state = np.load(s2, allow_pickle=True)

    # reading the first file
    path_random = './outputs_random/1000_episodes_random/'
    df_b = pd.read_csv(path_random + 'run_history_199.csv',
                       sep=';', index_col=0)
    # df_emb = pd.read_csv(path_random + 'run_history_emb_idx_199.csv',
    # sep=';', index_col=0)
    np_emb_state_b = np.load(
        path_random + 'emb_states_199.npy', allow_pickle=True)
    # np_emb_next_state = np.load(path_random + 'next_states_199.npy',
    # allow_pickle=True)

    skeeped_files = []
    # adding all the others

    list1 = list(range(399, 999, 200))
    for i in list1:
        s1 = path_random + 'run_history_' + str(i) + '.csv'
        df_ = pd.read_csv(s1, sep=';', index_col=0)
        s2 = path_random + 'emb_states_' + str(i) + '.npy'
        np_emb_state_ = np.load(s2, allow_pickle=True)
        if np_emb_state_.shape[0] != df_.shape[0]:
            print('skeeping file %d' % i)
            skeeped_files.append(s1)
            skeeped_files.append(s2)
        else:
            df_b = pd.concat([df_b, df_], axis=0, ignore_index=True)
            np_emb_state_b = np.concatenate((np_emb_state_b, np_emb_state_))

    return df, np_emb_state, df_b, np_emb_state_b


def add_expected_reward_to_df(df, total_episodes):
    discount = 0.99  # based in the PPO original paper, default discount

    for ep in total_episodes:
        df_ = df[df['episode'] == ep].copy()
        df_.sort_values(by=['step'], ascending=False, inplace=True)
        cum_reward = 0.0
        j = 0
        for i, step in df_.iterrows():
            if j == 0:
                cum_reward = step.reward
            else:
                cum_reward = cum_reward + discount * step.reward
            df.at[i, 'exp_reward'] = cum_reward
            j += 1
    return df


def dm_oppe(args, env, ppo_policy, df_b, np_emb_state_b,
            total_episodes, total_episodes_b, J_eps, model):

    print('Computing DM OPPE')
    num_experiments = 10
    batch_size = int(len(total_episodes_b) / num_experiments)
    actions = range(0, 26)
    prob_random = 1. / env.action_space.n
    a_t = torch.Tensor(actions).to(torch.int64)
    actions_one_hot = torch.nn.functional.one_hot(a_t)

    errors = []

    for exp in range(num_experiments):
        print("Starting experiment %d of %d" % (exp, num_experiments))
        episodes = np.random.choice(total_episodes_b, batch_size)
        values = []
        for ep in episodes:
            df_ = df_b[df_b['episode'] == ep].copy()

            # for i,step in df_.iterrows():
            # we only need the first state
            # step = df_.iloc[0]
            # global index
            i = df_.iloc[0].name

            probs = []
            q_estimations = []
            # selecting the state
            obs = torch.from_numpy(np.stack(np_emb_state_b[i]))
            obs = torch.unsqueeze(obs, 0)
            # summing up all the actions
            softmax = torch.nn.Softmax(dim=1)
            for action in actions:
                # action = step['action']
                a_t = actions_one_hot[action]
                # creating a batch of 1 element
                a_t = torch.unsqueeze(a_t, 0)
                t_x = torch.cat([obs, a_t], dim=1)
                x = torch.nn.functional.normalize(t_x)
                output = model(x)
                q_estimations.append(output[0].detach().numpy())

                if args.agent_type == 'ppo':
                    logits, _ = ppo_policy.model(
                                                {"obs": obs})
                    probs_ = softmax(logits)
                    prob_ = probs_[0][action]
                    probs.append(prob_.detach().numpy())
                elif args.agent_type == 'random':
                    probs.append(prob_random)

            values.append(np.squeeze(np.dot(np.transpose(
                np.array(q_estimations)),
                np.expand_dims(np.array(probs), axis=1))))

        dm_oppe = np.array(values).mean()
        print("Experiment %d of %d: Average DM OPPE for %s %.8f" %
              ((exp+1), num_experiments, args.agent_type, dm_oppe))
        error = np.square(dm_oppe - J_eps)
        print("Experiment %d of %d: Squared Error for %s %.8f" %
              ((exp+1), num_experiments, args.agent_type, error))
        errors.append(error)

    rmse = np.sqrt(np.array(errors).mean())
    std = np.array(errors).std()
    print("RSME DM OPPE for %d Experiments of %d episodes: %.8f" %
          (num_experiments, batch_size, rmse))
    print("STD DM OPPE for %d Experiments of %d episodes: %.8f" %
          (num_experiments, batch_size, std))


def is_ppo(args, env, ppo_policy, df_b, np_emb_state_b,
           total_episodes_b, J_eps):
    print('Computing IS OPPE')
    num_experiments = 10
    batch_size = int(len(total_episodes_b) / num_experiments)
    prob_behavior = 1. / env.action_space.n

    print('Calculating cum_reward of every episode, behavior policy')
    df_b = add_expected_reward_to_df(df_b, total_episodes_b)

    errors = []

    softmax = torch.nn.Softmax(dim=1)
    for exp in range(num_experiments):
        print("Starting experiment %d of %d" % (exp, num_experiments))
        episodes = np.random.choice(total_episodes_b, batch_size)
        w_episodes = []
        for ep in episodes:
            df_ = df_b[df_b['episode'] == ep].copy()
            probs = []
            for i, step in df_.iterrows():

                # selecting the state
                obs = torch.from_numpy(np.stack(np_emb_state_b[i]))
                obs = torch.unsqueeze(obs, 0)
                action = step['action']
                logits, _ = ppo_policy.model(
                                            {"obs": obs})
                probs_ = softmax(logits)
                prob_ = probs_[0][action]
                probs.append(prob_.detach().numpy() / prob_behavior)

            w_episodes.append(np.prod(probs) * df_["exp_reward"].iloc[0])

        is_oppe = np.array(w_episodes).mean()
        print("Experiment %d of %d: Average IS OPPE for %s %.8f" %
              ((exp+1), num_experiments, args.agent_type, is_oppe))
        error = np.square(is_oppe - J_eps)
        print("Experiment %d of %d: Squared Error for %s %.8f" %
              ((exp+1), num_experiments, args.agent_type, error))
        errors.append(error)

    rmse = np.sqrt(np.array(errors).mean())
    std = np.array(errors).std()
    print("RSME IS OPPE for %d Experiments of %d episodes: %.8f" %
          (num_experiments, batch_size, rmse))
    print("STD IS OPPE for %d Experiments of %d episodes: %.8f" %
          (num_experiments, batch_size, std))


def snis_ppo(args, env, ppo_policy, df_b,
             np_emb_state_b, total_episodes_b, J_eps):
    print('Computing SNIS OPPE')
    num_experiments = 10
    batch_size = int(len(total_episodes_b) / num_experiments)
    prob_behavior = 1. / env.action_space.n

    print('Calculating cum_reward of every episode, behavior policy')
    df_b = add_expected_reward_to_df(df_b, total_episodes_b)

    errors = []
    softmax = torch.nn.Softmax(dim=1)
    for exp in range(num_experiments):
        print("Starting experiment %d of %d" % (exp, num_experiments))
        episodes = np.random.choice(total_episodes_b, batch_size)
        w_episodes = []
        w_s = []
        for ep in episodes:
            df_ = df_b[df_b['episode'] == ep].copy()
            probs = []
            for i, step in df_.iterrows():

                # selecting the state
                obs = torch.from_numpy(np.stack(np_emb_state_b[i]))
                obs = torch.unsqueeze(obs, 0)
                action = step['action']
                logits, _ = ppo_policy.model(
                                            {"obs": obs})
                probs_ = softmax(logits)
                prob_ = probs_[0][action]
                probs.append(prob_.detach().numpy() / prob_behavior)
                w_s.append(prob_.detach().numpy() / prob_behavior)

            w_episodes.append(np.prod(probs) * df_["exp_reward"].iloc[0])

        snis_oppe = np.array(w_episodes).mean() / np.array(w_s).mean()
        print("Experiment %d of %d: Average SNIS OPPE for %s %.8f" %
              ((exp+1), num_experiments, args.agent_type, snis_oppe))
        error = np.square(snis_oppe - J_eps)
        print("Experiment %d of %d: Squared Error for %s %.8f" %
              ((exp+1), num_experiments, args.agent_type, error))
        errors.append(error)

    rmse = np.sqrt(np.array(errors).mean())
    std = np.array(errors).std()
    print("RSME SNIS OPPE for %d Experiments of %d episodes: %.8f" %
          (num_experiments, batch_size, rmse))
    print("STD SNIS OPPE for %d Experiments of %d episodes: %.8f" %
          (num_experiments, batch_size, std))


def main(args):

    tf.compat.v1.enable_eager_execution()
    # ray.init(local_mode=args.local_mode)
    ray.init(local_mode=True)

    env_config_fpath = args.env_config
    # Read configurations
    with open(env_config_fpath, "rb") as f:
        env_config = json.load(f)

    payloads_fpath = os.path.join(os.path.dirname(
        __file__), 'gym_waf', 'data', env_config["payloads"])
    env_config["payloadsPath"] = payloads_fpath

    register_env("rl-waf", env_creator)

    env = env_creator(env_config)
    print('Number of actions: %d' % env.action_space.n)
    print('State space dimension: %s' % env.observation_space.shape)
    if args.agent_type == 'ppo':
        algo_config = {
            "framework": "torch",
            "env": "rl-waf",
            "env_config": env_config,
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "num_workers": 0,
            "log_level": "WARN",
            "horizon": 30
        }
        algo = PPO(algo_config)
        algo.restore("./ckpt_ppo_agent_torch/checkpoint_000006_torch")
        print('Checkpoint loaded')
        path_to_trajectories = './outputs_ppo'
        ppo_policy = algo.get_policy()
    else:
        path_to_trajectories = './outputs_random/1000_episodes_random/'

    # generating some trajectories and evaluating the trained policy
    df, np_emb_state_, df_b, np_emb_state_b = load_df(
        path_to_trajectories, args.agent_type)
    print("Loaded original trajectories: %d" % np_emb_state_.shape[0])
    print("Embeddings: %d" % df.shape[0])
    assert np_emb_state_.shape[0] == df.shape[0], 'number ' \
           'of steps in trajectories does not match'
    print("Loaded random trajectories %d" % np_emb_state_b.shape[0])
    print("Random embedding: %d" % df_b.shape[0])
    assert np_emb_state_b.shape[0] == df_b.shape[0], 'number ' \
        'of steps in trajectories does not match'

    total_episodes = list(df.episode.unique())
    print('Number of episodes loaded (real ppo policy) for evaluation: %d' %
          len(total_episodes))

    total_episodes_b = list(df_b.episode.unique())
    print('Number of episodes loaded (behavior policy) for evaluation: %d' %
          len(total_episodes_b))

    # calculating expected reward for episode

    print('Calculating cum_reward of every episode, evaluation policy')
    df = add_expected_reward_to_df(df, total_episodes)

    print(df.head())
    print('Calculating Total Real Value Function')

    J_eps = 0.0
    df_ = df.groupby('episode').first()
    J_eps = df_['exp_reward'].mean()
    print('Total Real Value function of %s Policy: %.8f' %
          (args.agent_type, J_eps))

    # loading q_regressor
    print('Loading regressor')
    model = SimpleNet()
    model.load_state_dict(torch.load('./q_regressor_chkp/q_regressor.pth'))
    model.eval()
    print('Q_Regressor loaded')

    dm_oppe(args, env, ppo_policy, df_b, np_emb_state_b, total_episodes,
            total_episodes_b, J_eps, model)
    is_ppo(args, env, ppo_policy, df_b, np_emb_state_b,
           total_episodes_b, J_eps)
    snis_ppo(args, env, ppo_policy,
             df_b, np_emb_state_b, total_episodes_b, J_eps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train RL agent on WAF-Brain Environment with RLLib")
    parser.add_argument("--env-config", type=str,
                        help="Path to configuration file of the envionment.",
                        default="/home/azureuser/cloudfiles/code/Users/Enrique.Mora/rl4pentest/src/config/env.json")
    parser.add_argument("--local-mode", action="store_true",
                        help="Init Ray in local mode for easier debugging.")
    parser.add_argument("--agent_type", default="ppo")
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    main(args)
