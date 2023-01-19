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


if __name__ == '__main__':

    # init env
    MAXTURNS = 50

    # path to the list of sql injection payloads
    DATASET = os.path.join(os.path.dirname(__file__), 'gym_waf', 'data',
                           'sqli-1waf.csv')
    N_ITER = 30
    CHECKPOINT_ROOT = '/Users/ESMoraEn/repositories/rl-waf/checkpoints'

    sw = SummaryWriter('random_agent')

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
                                       'mutation_num', 'action', 'reward'])

    for i in range(500):
        print('Staring episode {}'.format(i))
        state = env.reset()
        score = 0
        mutation_num = 0
        while True:
            action = np.random.randint(action_space_size)
            print('Applying action {}'.format(action))
            state, reward, done, env_info = env.step(action)
            states.append(state)
            mutations.append(env_info['payload'])
            mutation_num += 1
            # print('Mutation number: {} - Reward: {}'.format(mutation_num,
            #                                               reward))

            # print_env_info(env_info)
            next_state = state
            score += reward
            state = next_state
            df_results = df_results.append(pd.Series([i, env.orig_payload,
                                                      mutation_num,
                                                      action, reward],
                                                     index=df_results.columns),
                                           ignore_index=True)
            # print(df_results)
            if done:
                if env_info['win']:
                    print('We have a winner!')
                    winner_infos.append(env_info)
                break
        scores.append(score)
        print("Score: {}".format(score))
        sw.add_scalar('reward', score, i)
        sw.add_scalar('num_mutations', mutation_num, i)
    df_results.to_csv('run_results.csv')

    # with open(f'./logs/embeddings/feature_vecs.tsv', 'w') as fw:
    #     csv_writer = csv.writer(fw, delimiter='\t')
    #     csv_writer.writerows(states)
    # with open(f'./logs/embeddings/metadata.tsv', 'w') as file:
    #     for mutation in mutations:
    #         file.write(f'{mutation}\n')

    print('Avg. score after episodes {}'.format(np.mean(scores)))
    print('Fin')
