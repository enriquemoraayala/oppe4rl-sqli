import os
from gym_waf.envs.waf_brain_env import WafBrainEnv
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

MAXTURNS = 30
DATASET = os.path.join(os.path.dirname(__file__), 'gym_waf','data', 'sqli-1waf.csv')

#loads all the sql injection payloads from the list and evaluate it with WafBrain
if __name__ == '__main__':

	wb = WafBrainEnv( DATASET, MAXTURNS) 
	wb._load_payloads(DATASET)

	for payload in wb.payload_list:
		res = wb.interface.get_score(payload)
		print("WafBrain result: ", res)
