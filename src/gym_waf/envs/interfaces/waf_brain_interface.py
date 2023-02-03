from .base_interface import LocalInterface, ClassificationFailure
from waf_brain.inferring import process_payload
from keras.models import load_model
from termcolor import colored
import os
import logging

os.makedirs("./logs", exist_ok=True)
logging.basicConfig(filename='./logs/waf_brain_env_logs.log',level=logging.DEBUG)

class WafBrainInterface(LocalInterface):
    def __init__(self, score_threshold=0.6):
        super().__init__()
        model_path = os.path.join(os.path.dirname(__file__),
                                  '../../models/waf-brain.h5')
        self.model = load_model(model_path)
        self.score_threshold = score_threshold

    def get_score(self, payload, check_weights: bool=False):
        result = process_payload(self.model, 'q', [payload], check_weights)
        logging.debug("WAF-BRAIN RESULT: {}".format(colored(repr(result), 'blue')))
        if result is None:
            raise ClassificationFailure('WAF-Brain classifier returns None')
        score = result['score']
        is_sqli = score > self.score_threshold
        return score, is_sqli
