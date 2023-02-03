from gym_waf.envs.features.tokenizer.tokenizer import TokenizerTK
import os
import tensorflow as tf


tf.compat.v1.disable_eager_execution()

MAXTURNS = 30
DATASET = os.path.join(os.path.dirname(__file__), 'gym_waf','data', 'sqli-1waf.csv')

# testing the tokenizer with various payloads
if __name__ == '__main__':

	tk = TokenizerTK()

	print("----------------------")
	payload = "a'\nor 1=1; -- \nt"
	payload1 = "a' or 1=1;\n-- "
	payload1 = "a' || 1=1;-- "
	print(payload)
	fv = tk._preprocess_input_query(payload)
	print(fv)

