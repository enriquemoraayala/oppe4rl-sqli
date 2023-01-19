import argparse
import json
import os
from azureml.core import Run
import torch
import numpy as np
import pandas as pd
from scipy import stats
from tensorboardX import SummaryWriter
from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
from gym_waf.envs.waf_brain_env import WafBrainEnv


sw = SummaryWriter('outputs_random/summary')


def main(args) -> None:
    run = Run.get_context(allow_offline=True)

    if args.from_config_files:
        env_config_fpath = args.env_config
        agent_config_fpath = args.agent_config
        # Read configurations
        with open(env_config_fpath, "rb") as f:
            env_config = json.load(f)
        with open(agent_config_fpath, "rb") as f:
            agent_config = json.load(f)
        payloads_fname = env_config["payloads"]
        max_steps = env_config["steps"]
        feature_extractor = env_config["featureExtractor"]
        agent_name = agent_config["agent"].lower()
        seed = agent_config.get("seed", None)
        episodes = agent_config["episodes"]
        allow_payload_actions = env_config["allowPayloadActions"] is True
        reward_name = env_config["reward"]
        reward_win_val = env_config["rewardWin"]
        strategy = env_config["strategy"]
        score_threshold = env_config["score_threshold"]
    else:
        payloads_fname = args.payloads
        max_steps = args.steps
        feature_extractor = args.feature_extractor
        agent_name = args.agent
        seed = args.seed
        episodes = args.episodes
        allow_payload_actions = args.allow_payload_actions == "true"
        reward_name = args.reward
        reward_win_val = args.reward_win
        strategy = args.strategy

    payloads_fpath = os.path.join(os.path.dirname(
        __file__), 'gym_waf', 'data', payloads_fname)
    env = WafBrainEnv(payloads_file=payloads_fpath, maxturns=max_steps, feature_extractor=feature_extractor,
                      payload_actions=allow_payload_actions, reward_name=reward_name, reward_win_val=reward_win_val,
                      score_threshold=score_threshold)
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.shape
    run.tag("envActionSpaceSize", action_space_size)
    run.tag("envStateSpaceSize", state_space_size)
    run.tag("envFeatureExtractor", feature_extractor)
    run.tag("envMaxSteps", max_steps)
    run.tag("envPayloads", payloads_fname)
    run.tag("envAllowPayloadActions", allow_payload_actions)
    run.tag("rewardName", reward_name)
    run.tag("rewardWinValue", reward_win_val)
    run.tag("envStrategy", strategy)

    eps_start = 1
    eps_end = 0.01
    eps_decay = 0.995
    scores = []
    eps = eps_start
    run.tag("agent", agent_name)
    run.tag("epsDecay", eps_decay)
    if agent_name == "random":
        agent = RandomAgent(state_space_size[0], action_space_size, seed=seed)
    elif agent_name == "dqn":
        agent = DQNAgent(state_space_size[0], action_space_size, seed=seed)
    else:
        pass

    if strategy == "first":
        strategy_i = 0
    elif strategy == "random":
        strategy_i = 1
    elif strategy == "all":
        strategy_i = 2

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
    episode_wins = [0] * episodes 
    for i_episode, episode in enumerate(range(1, episodes + 1)):
        state = env.reset()
        score = 0
        for i_step, step in enumerate(range(1, max_steps + 1)):
            action = agent.act(state, eps)
            next_state, reward, done, env_info = env.step(action, strategy_i)
            agent.step(state, action, reward, next_state, done)
            df_results = pd.concat(
                [df_results,
                 pd.DataFrame([[episode, step, env.orig_payload, env_info['previous_payload'],
                                action, env_info["payload"],
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
            state = next_state
            score += reward
            print(f'Episode: {episode} - Step: {step} - Action: {action} - Reward: {reward}')
            if done:
                if env_info['win']:
                    print('\n We have a winner!: ')
                    print('\n Episode {}\tTotal Average Score: {:.2f} \n'
                          .format(episode, np.mean(scores)), end="")
                    episode_wins[i_episode] = 1
                else:
                    episode_wins[i_episode] = 0
                run.log("winProportion", sum(episode_wins[:(i_episode + 1)]) / (i_episode + 1))
                episode_wins_last_50 = episode_wins[max(0, i_episode + 1 - 50):(i_episode + 1)]
                run.log("winProportionLast50", sum(episode_wins_last_50) / len(episode_wins_last_50))
                break
        scores.append((score/step))               # save most recent score (average)
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f} \n'
              .format(episode, np.mean(scores)), end="")
        sw.add_scalar('reward', score, episode)
        sw.add_scalar('num_mutations', step, episode)
        sw.add_scalar('Avg. last episodes reward', np.mean(scores))
        if episode % 50 == 0 and agent_name == "dqn":
            print('\rEpisode {}\tAverage Score: {:.2f}'
                  .format(episode, np.mean(scores)))
            torch.save(agent.qnetwork_local.state_dict(),
                       "agent_checkpoint.pth")
        os.makedirs("outputs", exist_ok=True)
        try:
            df_results.to_csv('outputs/run_history.csv', sep=';')
            df_results_emb.to_csv('outputs/run_history_emb.csv', sep=';')
        except Exception as e:
            print(e)

    ## METRICS ##
    history_by_episode = df_results.groupby(["episode"]).agg(
        original_payload=pd.NamedAgg(column="original_payload", aggfunc=lambda x: stats.mode(x)[0][0]),
        num_steps=pd.NamedAgg(column="step", aggfunc="max"),
        reward_avg=pd.NamedAgg(column="reward", aggfunc=np.mean),
        total_score=pd.NamedAgg(column="reward", aggfunc="sum"),
        win=pd.NamedAgg(column="win", aggfunc="sum")
    )
    history_by_payload = history_by_episode.groupby("original_payload").agg(
        num_eps=pd.NamedAgg(column="original_payload", aggfunc="count"),
        num_win=pd.NamedAgg(column="win", aggfunc="sum"),
        prop_win=pd.NamedAgg(column="win", aggfunc="mean"),
    ).sort_values("prop_win", ascending=False)
    log_dataframe(run, history_by_payload.reset_index(), "payload_wins")

    winners = history_by_episode[history_by_episode["win"] == 1]
    winners_stats = winners.groupby("original_payload").agg(
        num_wins=pd.NamedAgg(column="win", aggfunc="sum"),
        min_steps_win=pd.NamedAgg(column="num_steps", aggfunc="min"),
        mean_steps_win=pd.NamedAgg(column="num_steps", aggfunc="mean"),
        max_steps_win=pd.NamedAgg(column="num_steps", aggfunc="max"),
        std_steps_win=pd.NamedAgg(column="num_steps", aggfunc="std")
    ).sort_values("mean_steps_win", ascending=True)
    log_dataframe(run, winners_stats.reset_index(), "winner_payload_stats")
    return scores


def log_dataframe(run, df: pd.DataFrame, name: str) -> None:
    max_cols = 15
    max_rows = 10e+6
    max_rows_batch = 250
    if df.shape[0] > max_rows or df.shape[1] > max_cols:
        print(f"Metric {name} not logged, shape {df.shape} is over the limit.")
    for i_batch in range(0, df.shape[0], max_rows_batch):
        run.log_table(
            name,
            df.iloc[i_batch:(i_batch + max_rows_batch)].to_dict("list"))
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train RL agent on WAF-Brain Environment")
    parser.add_argument("--from-config-files", action="store_true",
                        help="Whether config files are used for configuration. Parameters not coming from config files will be ignored.")
    parser.add_argument("--env-config", type=str, help="Path to configuration file of the envionment.",
                        default="config/env.json")
    parser.add_argument("--agent-config", type=str, help="Path to configuration file of the agent.",
                        default="config/agent.json")
    parser.add_argument("--episodes", type=int, help="Number of episodes.",
                        default=1000)
    parser.add_argument("--steps", type=int, help="Maximum number of steps in each episode.",
                        default=30)
    parser.add_argument("--payloads", type=str, help="Name of the payloads file (in the target folder).",
                        default="sqli-1waf.csv")
    parser.add_argument("--start-with-payload", type=str, help="Episode starting with a random payload or empty payload.",
                        choices=["random", "empty"],
                        default="random")
    parser.add_argument("--allow-payload-actions", type=str, help="Whether selecting an initial payload is in action state.",
                        choices=["true", "false"],
                        default=True)
    parser.add_argument("--feature-extractor", type=str, help="Name of the feature extractor for SQL payloads.",
                        choices=["SqlSimpleFeatureExtractor",
                                 "SqlTermProportionFeatureExtractor", "SqlEmbedFeatureExtractor"],
                        default="SqlEmbedFeatureExtractor")
    parser.add_argument("--reward", type=str, help="Reward used by the environment.",
                        choices=["original", "probability", "binary"],
                        default="original")
    parser.add_argument("--reward-win", type=float, help="Reward of the environment when agent wins.",
                        default=0)
    parser.add_argument("--agent", type=str, help="Name of the agent",
                        choices=["random", "dqn"],
                        default="dqn")
    parser.add_argument("--strategy", type=str, help="Kind of strategy",
                choices=["first", "random", "all"],
                default="all")
    parser.add_argument("--seed", type=int,
                        help="Set random seed.", default=None)
    args = parser.parse_args()
    main(args)
