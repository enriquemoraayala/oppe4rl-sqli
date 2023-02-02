from termcolor import colored
import logging

from gym_waf.envs.interfaces import ClassificationFailure
from .waf_env import WafEnv

from gym_waf.envs.controls import sqlfuzzer as manipulate
ACTION_LOOKUP = {i: act for i, act in enumerate(
    manipulate.strategies)}


logging.basicConfig(filename='./logs/waf_brain_env_logs.log',
                    level=logging.DEBUG)


class WafScoreEnv(WafEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, payloads_file, maxturns=20, score_threshold=0.1, reward_name="original",
                 use_diff_reward=False, feature_extractor=None, payload_actions=False, reward_win_val: float=0.0):
        super().__init__(payloads_file, maxturns=maxturns, reward_win_val=reward_win_val,
                         feature_extractor=feature_extractor, payload_actions=payload_actions)
        self.score_threshold = score_threshold
        self.use_diff_reward = use_diff_reward
        self.reward = None
        self.score = None
        self.reward_name = reward_name
        self.reward_win_val = reward_win_val #upper threshold in reward
        self.total_reward_win_val = reward_win_val 

    def _get_score(self, payload):
        raise NotImplementedError("_get_score not implemented")

    def _check_sqli(self, payload):
        try:
            score, is_sqli = self._get_score(payload)
            logging.debug("Payload: {} - Score: {} - is_sqli: {}".format(self.payload, score, is_sqli))
        except ClassificationFailure:
            logging.warning("Failed to classify payload: {}".format(colored(repr(self.payload), 'red')))
            score = 0.01
            is_sqli = False
        self.score = score
        return is_sqli

    def step(self, action_index, strategy_index=2):
        assert self.orig_payload is not None, "please reset() before step()"

        self.turns += 1
        self.previous_payload = self.payload
        self.previous_observation = self.feature_extractor.extract(self.previous_payload)
        self._take_action(action_index, strategy_index)

        self.observation = self.feature_extractor.extract(self.payload)

        logging.debug("mutated payload: {}".format(self.payload))
        #logging.debug("state: {}".format(colored(repr(self.observation),
        #                                         'yellow')))

        win = False
        # get reward
        if not self._check_sqli(self.payload):  # Call to self._check_sqli sets self.score
            # we win!
            episode_over = True
            win = True
            logging.debug("WIN with payload: {}".format(self.payload))
        elif self.turns >= self.maxturns:
            # out of turns :(
            episode_over = True
            logging.debug("Finished with payload: {}".format(self.payload))
        else:
            episode_over = False

        if self.reward_name == "original":
            if win:
                new_reward = self.total_reward_win_val
            else:
                # caution, score between 0 and 1!!
                # new_reward = self.score_threshold * 10. / max(self.score, self.score_threshold)
                new_reward = self.score_threshold * self.reward_win_val / max(self.score, self.score_threshold)

            logging.debug("new_reward: {}".format(colored(repr(new_reward),
                                                        'yellow')))
            if self.use_diff_reward:
                if self.reward is None:
                    self.reward = old_reward = new_reward
                else:
                    old_reward = self.reward
                    self.reward = new_reward
                step_reward = new_reward - old_reward - self.turn_penalty
            else:
                step_reward = self._process_reward(new_reward)
        elif self.reward_name == "probability":  # step reward is negative with the injection detection probability
            new_reward = -self.score
            step_reward = new_reward
        elif self.reward_name == "binary":  # step reward is -1 if injection detected 'reward_win' if succeeded
            new_reward = -1 if win is False else self.reward_win_val
            step_reward = new_reward
        elif self.reward_name == "oppe4rl":
            if win:
                new_reward = self.total_reward_win_val
                step_reward = new_reward
            else:
                new_reward = -self.score
                step_reward = self._process_reward(new_reward)

        logging.debug("step_reward: {}".format(step_reward))

        if episode_over:
            logging.debug("episode is over: reward = {}!".format(step_reward))

        return self.observation, step_reward, episode_over, \
            {"win": win, "original": self.orig_payload,
             "payload": self.payload, "history": self.history, "previous_payload": self.previous_payload,
             "emb_payload": self.observation, "emb_prev_payload": self.previous_observation}
