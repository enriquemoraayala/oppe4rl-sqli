import numpy as np
import gym
import random
import logging
from gym_waf.envs.features import SqlSimpleFeatureExtractor, SqlTermProportionFeatureExtractor, SqlEmbedFeatureExtractor
from termcolor import colored

from gym_waf.envs.controls import sqlfuzzer as manipulate
ACTION_LOOKUP = {i: act for i, act in enumerate(
    manipulate.strategies)}
    
MUTATION_LOOKUP = {i: act for i, act in enumerate(
    manipulate.place_mutation)}

SEED = 0
logging.basicConfig(filename='./logs/waf_env_logs.log', level=logging.DEBUG)


class WafEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, payloads_file, maxturns=20, turn_penalty=0.1, reward_win_val=10., feature_extractor:str="SqlTermProportionFeatureExtractor",
                 payload_actions=False):
        """
        Base class for WAF env
        :param payloads: a list of payload strings
        :param maxturns: max mutation before env ends
        """
        self.payload_list = None
        self._load_payloads(payloads_file)
        self.payload_actions = payload_actions
        if payload_actions is True:
            self.action_space = gym.spaces.Discrete(len(ACTION_LOOKUP) + len(self.payload_list))
        else:
            self.action_space = gym.spaces.Discrete(len(ACTION_LOOKUP))
        self.maxturns = maxturns
        self.feature_extractor = self._get_feature_extractor(feature_extractor)
        logging.debug("Feature vector shape: {}".format(self.feature_extractor.shape))
        self.history = []
        self.max_reward = reward_win_val 
        self.min_reward = 0.0
        self.orig_payload = None
        if feature_extractor == 'SqlEmbedFeatureExtractor':
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.feature_extractor.shape, dtype=np.float32)    
        else:
            self.observation_space = gym.spaces.Box(low=0., high=1., shape=self.feature_extractor.shape, dtype=np.float32)
        self.turn_penalty = turn_penalty

        self.payload = None
        self.observation = None
        self.turns = 0

    def _load_payloads(self, filepath):
        try:
            with open(filepath, 'r', encoding='UTF-8') as f:
                self.payload_list = f.read().splitlines()
                logging.debug("{} payloads loaded".format(len(self.payload_list)))
        except OSError as e:
            logging.error("failed to load dataset from {}".format(filepath))
            raise
    
    def _get_feature_extractor(self, name: str):
        if name == "SqlSimpleFeatureExtractor":
            return SqlSimpleFeatureExtractor()
        elif name == "SqlTermProportionFeatureExtractor":
            return SqlTermProportionFeatureExtractor()
        elif name == "SqlEmbedFeatureExtractor":
            print("YYY")
            return SqlEmbedFeatureExtractor()
        else:
            raise(Exception(f"{name} is not a valid feature extractor name"))

    def step(self, action_index):
        raise NotImplementedError("_step not implemented")

    def _check_sqli(self, payload):
        raise NotImplementedError("_check_sqli not implemented")

    def _take_action(self, action_index, strategy_index):
        #assert action_index < len(ACTION_LOOKUP)
        assert action_index < self.action_space.n
        if action_index < len(ACTION_LOOKUP):
            action = ACTION_LOOKUP[action_index]
            mutation = MUTATION_LOOKUP[strategy_index]
            logging.debug(action.__name__)
            self.history.append(action)
            self.payload = action(self.payload, mutation, seed=SEED)
        else:
            payload_idx = action_index - len(ACTION_LOOKUP)
            payload = self.payload_list[payload_idx]
            self.history.append(payload)
            self.payload = payload
            logging.debug(f"Set initial payload {payload}")

    def _process_reward(self, reward):
        reward = reward - (self.turns - 1) * self.turn_penalty  # encourage fewer turns
        # reward = max(min(reward, self.max_reward), self.min_reward)
        return reward

    def reset(self, payload: str=None):
        self.turns = 0

        #while True:     # until find one that is SQLi by the interface
        if payload is None:
            payload = random.choice(self.payload_list)
        else:
            payload = payload
        #    if self._check_sqli(payload):
        self.orig_payload = self.payload = payload
        #        break
        #logging.debug("skipping payload that is not sql injection: {}".format(colored(repr(payload), 'red')))
       
       # logging.debug("reset payload: {}".format(self.payload))
        logging.debug("reset payload: {}".format(colored(repr(self.payload), 'green')))
        self.observation = self.feature_extractor.extract(self.payload)

        return self.observation

    def reset_with_payload(self, payload):
        self.turns = 0

        #if self._check_sqli(payload):
        self.orig_payload = self.payload = payload

        #logging.debug("skipping payload that is not sql injection: {}".format(colored(repr(payload), 'red')))
    

        logging.debug("reset payload: {}".format(colored(repr(self.payload), 'green')))
        self.observation = self.feature_extractor.extract(self.payload)

        return self.observation    

    def render(self, mode='human', close=False):
        pass
