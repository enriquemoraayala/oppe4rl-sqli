from .waf_score_env import WafScoreEnv
from .interfaces import WafBrainInterface


class WafBrainEnv(WafScoreEnv):
    def __init__(self, payloads_file, maxturns=20, score_threshold=0.1,
                 use_diff_reward=False, feature_extractor: str="SqlTermProportionFeatureExtractor", 
                 payload_actions=False, reward_name: str="original", reward_win_val: float=0.0):
        super().__init__(payloads_file, maxturns=maxturns,
                         use_diff_reward=use_diff_reward, feature_extractor=feature_extractor, payload_actions=payload_actions,
                         reward_name=reward_name, reward_win_val=reward_win_val, score_threshold=score_threshold)
        self.interface = WafBrainInterface(score_threshold=score_threshold)

    def _get_score(self, payload):
        return self.interface.get_score(payload)
