from gym import logger
import os
from gym_waf.envs.waf_brain_env import WafBrainEnv
import logging
from ray.rllib.agents.ppo import PPOTrainer
import ray.rllib.agents.ppo as ppo
import ray
from ray.tune.registry import register_env
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

MAXTURNS = 10
DATASET = os.path.join(os.path.dirname(__file__), 'gym_waf','data', 'sqli-1waf.csv')

class Result: 
	def __init__(self, original,mutated, win): 
		self.original = original 
		self.win = win
		self.mutated = mutated


def isDuplicate(list, original):
	duplicate = False
	for elem in list:
		if elem.original == original:
			duplicate = True
			break
	return duplicate

def winCount(list, win):
	cnt = 0
	for elem in list:
		if elem.win == win:
			cnt += 1
	return cnt	


# evaluation of trained agent
if __name__ == '__main__':
	resultList = []
	
	ray.init(_node_ip_address="0.0.0.0")

	logging.basicConfig(filename="myPPOlog.log")
	logger = logging.getLogger()
	logger.setLevel('DEBUG')

	def env_creator(env_config):
		return WafBrainEnv( DATASET, MAXTURNS)  

	register_env("my_env", env_creator)

	config = ppo.DEFAULT_CONFIG.copy()
	#turn off the exploration
	config["explore"] = False
	config["env"] = "my_env"
	#define the model layers
	config["model"]["fcnet_hiddens"] = [512,512,256 ]
	agent = PPOTrainer(config)

	#restore the saved model
	agent.restore('/Users/ESMoraEn/repositories/rl-waf/gym-waf-results/WAFBRAIN-PPO/Original-actions/checkpoint_000180/checkpoint-180')
	env = env_creator("")

	episode_count = 100
	reward = 0
	done = False

	env._load_payloads(DATASET)

	for payload in env.payload_list:
		#only try to break payloads that are initially identified by waf as sql injections
		if env._check_sqli(payload):
			print("Original {}".format(payload))

			#try to mutate each payload 30 times, or continue to next one if successfull 
			for i in range(30):
				ob = env.reset_with_payload(payload)
				w = False
				while True:
					action= agent.compute_action(observation=ob)
					ob, reward, done, info = env.step(action)
					if done:
						resultList.append(Result(info["original"],info["payload"], info["win"]))
						w = info["win"]

						#if the episode finished with the win, print the number of iteration and 
						# mutated paylaod
						if w == True:
							print("Iteration {}".format(i))
							print("Mutated {}".format(info["payload"]))
						break
				if w == True:
					break
		else:
			print("Not a sqli based on waf brain: ", payload)


	print("List len: {}".format(len(env.payload_list)))

	print("Win {}".format(winCount(resultList, True)))

	env.close()			


