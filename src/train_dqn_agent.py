import os
import torch
import numpy as np
import pandas as pd
from gym_waf.envs.waf_brain_env import WafBrainEnv
from collections import deque
from tensorboardX import SummaryWriter
from agents.dqn_agent import DQNAgent

sw = SummaryWriter('outputs/summary')


def waf_dqn(env, ckp_path, n_episodes=150,
            max_t=20,
            eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy
        action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode)
        for decreasing epsilon
    """
    scores = []
    scores_window = deque(maxlen=10)  # last 100 scores
    eps = eps_start
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.shape
    print('Action space size %i' % action_space_size)
    print('State space size %i' % state_space_size)
    scores = []
    winner_infos = []
    states = []
    mutations = []
    df_results = pd.DataFrame(columns=['eps_id', 'original_payload',
                                       'mutation_num', 'action', 'payload','reward', 'win'])

    agent = DQNAgent(state_space_size[0], action_space_size, seed=0)

    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset(payload="")
        score = 0
        mutation_num = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, env_info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            states.append(state)
            mutations.append(env_info['payload'])
            mutation_num += 1
            score += reward
            print(f'Episode: {i_episode} - Mutation number: {mutation_num} - Reward: {reward}')
            df_results = pd.concat(
                [df_results,
                 pd.DataFrame([[i_episode, env.orig_payload, mutation_num, action, env_info["payload"], reward, env_info['win']]],
                              columns=df_results.columns)],
                axis=0,
                ignore_index=True)
            if done:
                if env_info['win']:
                    print('\n We have a winner!: ')
                    print('\n Episode {}\tAverage Score: {:.2f} \n'
                          .format(i_episode, np.mean(scores_window)), end="")
                    winner_infos.append(env_info)
                break

        scores_window.append(score)        # save most recent score
        scores.append(score)               # save most recent score
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'
              .format(i_episode, np.mean(scores_window)), end="")
        sw.add_scalar('reward', score, i_episode)
        sw.add_scalar('num_mutations', mutation_num, i_episode)
        sw.add_scalar('Avg. last episodes reward', np.mean(scores))
        if i_episode % 50 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'
                  .format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), ckp_path)
        os.makedirs("outputs", exist_ok=True)
        try:
            df_results.to_csv('outputs/run_dqn_results.csv')
        except Exception as e:
            print(e)
    return scores


def train_agent():
    # init env
    N_EPISODES = 500
    MAXTURNS = 30
    # path to the list of sql injection payloads
    DATASET = os.path.join(os.path.dirname(__file__), 'gym_waf', 'data',
                           'sqli-1waf.csv')
    N_ITER = 30
    FEATURE_EXTRACTOR = "SqlTermProportionFeatureExtractor"
    env = WafBrainEnv(DATASET, MAXTURNS, feature_extractor=FEATURE_EXTRACTOR)
    scores = waf_dqn(env, 'checkpoint_dqn_waf.pth', max_t=N_ITER, n_episodes=N_EPISODES)
    return scores


if __name__ == '__main__':
    train_agent()
