import os
import re
import numpy as np
import csv
import pandas as pd

from gym_waf.envs.waf_brain_env import WafBrainEnv
from tensorboardX import SummaryWriter


def print_env_info(env_info):
    print('Win: %s' % env_info['win'])
    print('Original: %s' % env_info['original'])
    print('Payload: %s' % env_info['payload'])
    print('History: %s' % env_info['history'])


def train_agent():
    # init env
    MAXTURNS = 30
    # path to the list of sql injection payloads
    DATASET = os.path.join(os.path.dirname(__file__), 'gym_waf', 'data',
                           'sqli-1waf.csv')
    CHECKPOINT_ROOT = './checkpoints'

    sw = SummaryWriter('outputs/summary')

    env = WafBrainEnv(DATASET, MAXTURNS)

    action_space_size = env.action_space.n
    state_space_size = env.observation_space.shape
    print('Action space size %i' % action_space_size)
    print('State space size %i' % state_space_size)
    print('Action sample:')
    print(env.action_space.sample())
    print('Observation sample:')
    print(env.observation_space.sample().shape)
    print(env.observation_space.sample())
    scores = []
    winner_infos = []
    states = []
    mutations = []
    df_results = pd.DataFrame(columns=['eps_id', 'original_payload',
                                       'step','state_t', 'action', 'state_t_plus_1', 'reward', 'win'])

    for i in range(500):
        print('Staring episode {}'.format(i))
        state = env.reset()
        score = 0
        mutation_num = 0
        while True:
            action = np.random.randint(action_space_size)
            print('Applying action {}'.format(action))
            state, reward, done, env_info = env.step(action, 2)
            states.append(state)
            mutations.append(env_info['payload'])
            mutation_num += 1
            print(f'Episode: {i} - Mutation number: {mutation_num} - Reward: {reward}')
            next_state = state
            score += reward
            state = next_state
            df_results = pd.concat(
                [df_results,
                 pd.DataFrame([i, env.orig_payload, mutation_num, state, action, env_info["payload"], reward, env_info['win']],
                              columns=df_results.columns)],
                axis=0,
                ignore_index=True)
            # print(df_results)
            if done:
                if env_info['win']:
                    print('We have a winner!')
                    winner_infos.append(env_info)
                try:
                    df_results.to_csv('run_results.csv')
                except Exception as e:
                    print(e)
                break
        scores.append(score)
        print("Score: {}".format(score))
        sw.add_scalar('reward', score, i)
        sw.add_scalar('num_mutations', mutation_num, i)
        try:
            df_results.to_csv('outputs/run_results.csv')
        except Exception as e:
            print(e)
    # with open(f'./logs/embeddings/feature_vecs.tsv', 'w') as fw:
    #     csv_writer = csv.writer(fw, delimiter='\t')
    #     csv_writer.writerows(states)
    # with open(f'./logs/embeddings/metadata.tsv', 'w') as file:
    #     for mutation in mutations:
    #         file.write(f'{mutation}\n')
    try:
        df_results.to_csv('outputs/run_results.csv')
    except Exception as e:
        print(e)
    print('Avg. score after episodes {}'.format(np.mean(scores)))
    print('Fin')


if __name__ == '__main__':
    train_agent()
    print(0)