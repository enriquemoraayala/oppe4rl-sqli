import argparse
import sys
from numpy import random
import gym
from gym import wrappers, logger
from gym.envs.registration import register
import os
import gym_waf
import logging
import re
from gym_waf.envs.controls import sqlfuzzer as manipulate


class Result: 
	def __init__(self, original,mutated, win): 
		self.original = original 
		self.win = win
		self.mutated = mutated

class RandomAgent(object):

	def __init__(self, action_space):
		self.action_space = action_space

	def act(self, observation, reward, done):
		return self.action_space.sample()

def winCount(list, win):
	cnt = 0
	for elem in list:
		if elem.win == win:
			cnt += 1
	return cnt	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=None)
	parser.add_argument('env_id', nargs='?', default='WafLibinj-v0', help='Select the environment to run')
	args = parser.parse_args()

	resultList = []

	# You can set the level to logger.DEBUG or logger.WARN if you
	# want to change the amount of output.
	logging.basicConfig(filename="mylog.log")
	logger = logging.getLogger()
	logger.setLevel('DEBUG')

	env = gym.make(args.env_id)

	env.seed(0)
	agent = RandomAgent(env.action_space)
	print('Observation space', env.observation_space.shape)
	print('Action space', env.action_space.shape)

	episode_count = 1000
	reward = 0
	done = False

	for i in range(episode_count):
		ob = env.reset()
		while True:
			action = agent.act(ob, reward, done)

			ob, reward, done, info = env.step(action)

			if done:
				resultList.append(Result(info["original"],info["payload"], info["win"]))
				break

	env.close()


	print("{}".format(len(resultList)))

	print("Win {}".format(winCount(resultList, True)))

	for res in resultList:
		if res.win == True:
			print("Original {}".format(res.original))
			print("Mutated {}".format(res.mutated))


