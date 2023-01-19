"""Strategies and fuzzer class module"""

from numpy import random
from functools import wraps
import re
from gym_waf.envs.controls.fuzz_utils import (
	num_tautology_ex,
	replace_random,
	filter_candidates,
	random_string,
	num_tautology,
	string_tautology,
	num_contradiction,
	string_contradiction,
	string_tautology_ex,
)

def selectReplace(payload, placeMutation, searchType, replacement, seed=None):
	if placeMutation == "first":
		pos = re.search(re.escape(searchType), payload)
		if not pos:
			return payload
		payload = payload[:pos.span()[0]] + replacement + payload[pos.span()[1]:]

	else:
		# positions = list(re.finditer(searchType, payload))
		positions = [m for m in re.finditer(re.escape(searchType), payload)]
		if not positions:
			return payload
		if placeMutation == "random":
			rng = random.RandomState(seed)
			pos = rng.choice(positions).span()
			payload = payload[:pos[0]] + replacement + payload[pos[1]:]
		elif placeMutation == "all":
			len_rep = 0
			for pos in positions:
				payload = payload[:pos.span()[0]+len_rep] + replacement + payload[pos.span()[1]+len_rep:]
				len_rep += len(replacement) - (pos.span()[1] - pos.span()[0])
	return payload

#/*fdsfsf*/ => /**/
def reset_inline_comments(payload: str, placeMutation, seed=None):   #if repeated cases takes one randomly
	"""Remove randomly chosen multi-line comment content.
	Arguments:
		payload: query payload string

	Returns:
		str: payload modified
	"""
	replacement = "/**/"
	searchType = r"/\*[^(/\*|\*/)]*\*/"
	payload = selectReplace(payload, placeMutation, searchType, replacement, seed)

	# print(payload)
	return payload

def logical_invariant(payload, placeMutation, seed=None): #takes the first one seen
	"""logical_invariant

	Adds an invariant boolean condition to the payload

	E.g., something OR False

	:param payload:
	"""
	rng = random.RandomState(seed)

	replacement = rng.choice(
	[
		# AND True
		" AND 1",
		" AND True",
		" AND " + num_tautology(),
		" AND " + string_tautology(),
		# OR False
		" OR 0",
		" OR False",
		" OR " + num_contradiction(),
		" OR " + string_contradiction(),
	]
	)

	if placeMutation == "first":
		pos = re.search("(#|-- )", payload)
		if not pos:
			return payload
		pos = pos.start()
		payload = payload[:pos] + replacement + payload[pos:]

	else:
		pos_list = list(re.finditer("(#|-- )", payload))
		# for match in re.finditer("(#|-- )", payload):
		# 	pos_list.append(match.span())

		if not pos_list:
			return payload

		if placeMutation == "random":
			# ind = list(range(len(pos_list)))
			rng = random.RandomState(seed)
			pos = rng.choice(pos_list).span()
			payload = payload[:pos[0]] + replacement + payload[pos[0]:]
		elif placeMutation == "all":
			len_rep = 0
			for pos in pos_list:
				payload = payload[:pos.span()[0]+len_rep] + replacement + payload[pos.span()[0]+len_rep:]
				len_rep += len(replacement)
	# print(payload)
	return payload

def change_num_tautologies(payload,placeMutation,index, seed=None):    #if repeated cases takes one randomly

	replacement = num_tautology_ex(index)
	searchType = r'((?<=[^\'"\d\wx])\d+(?=[^\'"\d\wx]))=\1'
	payload = selectReplace(payload, placeMutation, searchType, replacement, seed)
	# print(payload)
	return payload

def change_num_tautologies_EQUAL(payload, placeMutation, seed=None):  #if repeated cases takes one randomly
	return change_num_tautologies(payload,placeMutation, 0, seed)

def change_num_tautologies_NOTEQUAL(payload, placeMutation, seed=None):  #if repeated cases takes one randomly
	return change_num_tautologies(payload,placeMutation, 2, seed)

def change_num_tautologies_IN(payload, placeMutation, seed=None):  #if repeated cases takes one randomly
	return change_num_tautologies(payload,placeMutation, 5, seed)

def change_string_tautologies(payload, placeMutation, index, seed=None):  #if repeated cases takes one randomly

	replacement = string_tautology_ex(index)
	searchType = r'((?<=[^\'"\d\wx])\d+(?=[^\'"\d\wx]))=\1'
	payload = selectReplace(payload, placeMutation, searchType, replacement, seed)
	# print(payload)
	return payload

#just replacing with =, replacing = to LIKE should be part of other mutation
def change_string_tautologies_EQUAL1(payload, placeMutation, seed=None):  #if repeated cases takes one randomly
	return change_string_tautologies(payload, placeMutation, 0, seed)

def change_string_tautologies_EQUAL2(payload, placeMutation, seed=None):  #if repeated cases takes one randomly
	return change_string_tautologies(payload, placeMutation, 2, seed)


def change_string_tautologies_NOTEQUAL1(payload, placeMutation, seed=None):  #if repeated cases takes one randomly
	return change_string_tautologies(payload, placeMutation, 4, seed)

def change_string_tautologies_NOTEQUAL2(payload, placeMutation, seed=None):  #if repeated cases takes one randomly
	return change_string_tautologies(payload, placeMutation, 7, seed)



#replace a space with a multiline comment /**/
def spaces_to_comments(payload, placeMutation, seed=None):    #if repeated cases takes one randomly
	symbols = {" ": ["/**/"]}
	return swap_keywords_util(payload, placeMutation, symbols, seed)

#replace a multiline comment /**/ with a space
def comments_to_spaces(payload, placeMutation, seed=None):   #if repeated cases takes one randomly
	symbols = { "/**/": [" "]}
	return swap_keywords_util(payload, placeMutation, symbols, seed)
'''
def spaces_to_whitespaces_alternatives(payload, seed=None):
	rng = random.RandomState(seed)

	symbols = {
		" ": ["\t", "\n", "\f", "\v"],
		"\t": [" ", "\n", "\f", "\v"],
		"\n": ["\t", " ", "\f", "\v"],
		"\f": ["\t", "\n", " ", "\v"],
		"\v": ["\t", "\n", "\f", " "]
	}

	symbols_in_payload = filter_candidates(symbols, payload)

	if not symbols_in_payload:
		return payload

	# Randomly choose symbol
	candidate_symbol = rng.choice(symbols_in_payload)
	# Check for possible replacements
	replacements = symbols[candidate_symbol]
	# Choose one replacement randomly
	candidate_replacement = rng.choice(replacements)

	# Apply mutation at one random occurrence in the payload
	return replace_random(payload, candidate_symbol, candidate_replacement)
'''
def spaces_to_whitespaces_alternatives_sym(payload, placeMutation, symbols, seed=None):  #if repeated cases takes one randomly

		rng = random.RandomState(seed)
		symbols_in_payload = filter_candidates(symbols, payload)

		if not symbols_in_payload:
			return payload
		# Randomly choose symbol
		candidate_symbol = rng.choice(symbols_in_payload)
		# Check for possible replacements
		replacements = symbols[candidate_symbol]
		# Choose one replacement randomly
		replacement = replacements[0]
		searchType = candidate_symbol
		payload = selectReplace(payload, placeMutation, searchType, replacement, seed)

		return payload


def spaces_to_whitespaces_space_tab(payload, placeMutation, seed=None):  #if repeated cases takes one randomly
	symbols = {	" ": ["\t "]	}
	return spaces_to_whitespaces_alternatives_sym(payload, placeMutation, symbols, seed=None)

def spaces_to_whitespaces_space_newline(payload, placeMutation, seed=None):  #if repeated cases takes one randomly
	symbols = {	" ": ["\n "]	}
	return spaces_to_whitespaces_alternatives_sym(payload, placeMutation, symbols, seed=None)

def spaces_to_whitespaces_space_f(payload, placeMutation, seed=None):  #if repeated cases takes one randomly
	symbols = {	" ": ["\f "]	}
	return spaces_to_whitespaces_alternatives_sym(payload, placeMutation, symbols, seed=None)

def spaces_to_whitespaces_space_vert(payload, placeMutation, seed=None): #if repeated cases takes one randomly
	symbols = {	" ": ["\v "]	}
	return spaces_to_whitespaces_alternatives_sym(payload, placeMutation, symbols, seed=None)

def spaces_to_whitespaces_space_extraspace(payload, placeMutation, seed=None):  #if repeated cases takes one randomly
	symbols = {	" ": ["  "]	}
	return spaces_to_whitespaces_alternatives_sym(payload, placeMutation, symbols, seed=None)

def spaces_to_whitespaces_space_comment_newline(payload, placeMutation, seed=None):  #if repeated cases takes one randomly
	comment = random_string(7,False)
	sym = "#" + comment + "\n"
	symbols = {	" ": [sym]	}
	return spaces_to_whitespaces_alternatives_sym(payload, placeMutation, symbols, seed=None)

def random_case(payload, placeMutation, seed=None):
	rng = random.RandomState(seed)

	new_payload = []

	for c in payload:
		if rng.rand() > 0.5:
			c = c.swapcase()
		new_payload.append(c)

	return "".join(new_payload)

#  /**/ => /*fsdgerge*/
def multiline_comment_rewriting(payload, placeMutation, seed=None):  #if repeated cases takes one randomly

	replacement = random_string()
	searchType = r"/\*[^(/\*|\*/)]*\*/"
	payload = selectReplace(payload, placeMutation, searchType, replacement, seed)
	return payload

#  --erwtbr  => --erwtbr12
def singleline_comment_rewriting(payload, placeMutation, seed=None):
	if "#" in payload or "-- " in payload:
		return payload + random_string(2)
	else:
		return payload

# 1 => 0x1  (the number has to have space before and after)
def swap_int_to_hex_repr(payload, placeMutation, seed=None):

	rng = random.RandomState(seed)
	if placeMutation == "first":
		pos = re.search(r'(?<=[^\'"\d\wx])\d+(?=[^\'"\d\wx])', payload)
		if not pos:
			return payload
		candidate = payload[pos.span()[0] : pos.span()[1]]
		replacements = [hex(int(candidate))]
		replacement = rng.choice(replacements)
		payload = payload[:pos.span()[0]] + replacement + payload[pos.span()[1]:]

	else:
		candidates = list(re.finditer(r'(?<=[^\'"\d\wx])\d+(?=[^\'"\d\wx])', payload))
		if not candidates:
			return payload
		if placeMutation == "random":
			candidate_pos = rng.choice(candidates).span()
			candidate = payload[candidate_pos[0] : candidate_pos[1]]
			replacements = [hex(int(candidate))]
			replacement = rng.choice(replacements)
			payload = payload[:candidate_pos[0]] + replacement + payload[candidate_pos[1]:]
		elif placeMutation == "all":
			len_rep = 0
			for candidate_pos in candidates:
				candidate = payload[candidate_pos.span()[0]+len_rep : candidate_pos.span()[1]+len_rep]
				replacements = [hex(int(candidate))]
				replacement = rng.choice(replacements)
				payload = payload[:candidate_pos.span()[0]+len_rep] + replacement + payload[candidate_pos.span()[1]+len_rep:]
				len_rep += len(replacement) - (candidate_pos.span()[1] - (candidate_pos.span()[0]))
	# print(payload)
	return payload

# 1 => (select 1)  (the number has to have space before and after)
def swap_int_to_select_repr(payload, placeMutation, seed=None):

	rng = random.RandomState(seed)
	if placeMutation == "first":
		pos = re.search(r'(?<=[^\'"\d\wx])\d+(?=[^\'"\d\wx])', payload)
		if not pos:
			return payload
		candidate = payload[pos.span()[0] : pos.span()[1]]
		replacements = ["(SELECT {})".format(candidate)]
		replacement = rng.choice(replacements)
		payload = payload[:pos.span()[0]] + replacement + payload[pos.span()[1]:]

	else:
		candidates = list(re.finditer(r'(?<=[^\'"\d\wx])\d+(?=[^\'"\d\wx])', payload))
		if not candidates:
			return payload
		if placeMutation == "random":
			candidate_pos = rng.choice(candidates).span()
			candidate = payload[candidate_pos[0] : candidate_pos[1]]
			replacements = ["(SELECT {})".format(candidate)]
			replacement = rng.choice(replacements)
			payload = payload[:candidate_pos[0]] + replacement + payload[candidate_pos[1]:]
		elif placeMutation == "all":
			len_rep = 0
			for candidate_pos in candidates:
				candidate = payload[candidate_pos.span()[0]+len_rep : candidate_pos.span()[1]+len_rep]
				replacements = ["(SELECT {})".format(candidate)]
				replacement = rng.choice(replacements)
				payload = payload[:candidate_pos.span()[0]+len_rep] + replacement + payload[candidate_pos.span()[1]+len_rep:]
				len_rep += len(replacement) - (candidate_pos.span()[1] - (candidate_pos.span()[0]))

	# print(payload)
	return payload

"""
def swap_keywords(payload, seed=None):
	rng = random.RandomState(seed)

	symbols = {
		# OR
		"||": [" OR ", " || "],
		" || ": [" OR ", "||"],
		"OR": [" OR ", "||"],
		"  OR  ": [" OR ", "||", " || "],
		# AND
		"&&": [" AND ", " && "],
		" && ": ["AND", " AND ", " && "],
		"AND": [" AND ", "&&", " && "],
		"  AND  ": [" AND ", "&&"],
		# Not equals
		"<>": ["!=", " NOT LIKE "],
		"!=": [" != ", "<>", " <> ", " NOT LIKE "],
		# Equals
		" = ": [" LIKE ", "="],
		"LIKE": [" LIKE ", "="],
	}

	symbols_in_payload = filter_candidates(symbols, payload)

	if not symbols_in_payload:
		return payload

	# Randomly choose symbol
	candidate_symbol = rng.choice(symbols_in_payload)
	# Check for possible replacements
	replacements = symbols[candidate_symbol]
	# Choose one replacement randomly
	candidate_replacement = rng.choice(replacements)

	# Apply mutation at one random occurrence in the payload
	return replace_random(payload, candidate_symbol, candidate_replacement)
"""

def swap_keywords_OR(payload, placeMutation, seed=None):  #if repeated cases takes one randomly
	symbols = { "or": [" || "]	}
	return swap_keywords_util(payload, placeMutation, symbols, seed)

def swap_keywords_AND(payload, placeMutation, seed=None):  #if repeated cases takes one randomly
	symbols = {	"and": [" && "]	}
	return swap_keywords_util(payload, placeMutation, symbols, seed)

def swap_keywords_NOT_EQUAL_SYMB(payload, placeMutation, seed=None):  #if repeated cases takes one randomly
	symbols = {	"<>": [" != "]	}
	return swap_keywords_util(payload, placeMutation, symbols, seed)

def swap_keywords_NOTEQUAL_NOTLIKE(payload, placeMutation, seed=None):  #if repeated cases takes one randomly
	symbols = {	"<>": [" NOT LIKE "]}
	return swap_keywords_util(payload, placeMutation, symbols, seed)

def swap_keywords_EQUAL_LIKE(payload, placeMutation, seed=None): #if repeated cases takes one randomly
	symbols = {	"=": [" LIKE "]}
	return swap_keywords_util(payload, placeMutation, symbols, seed)


def swap_keywords_util(payload, placeMutation, symbols, seed=None):  #swaps keyword if repeated cases swaps one randomly

	rng = random.RandomState(seed)
	symbols_in_payload = filter_candidates(symbols, payload)
	if not symbols_in_payload:
		return payload
	# Randomly choose symbol
	candidate_symbol = rng.choice(symbols_in_payload)
	# Check for possible replacements
	replacements = symbols[candidate_symbol]
	# Choose one replacement randomly
	replacement = replacements[0]
	searchType = candidate_symbol
	payload = selectReplace(payload, placeMutation, searchType, replacement, seed)

	# print(payload)
	return payload


strategies = [
	spaces_to_comments,
	comments_to_spaces,
#	random_case,

	swap_keywords_EQUAL_LIKE,
	swap_keywords_NOTEQUAL_NOTLIKE,
	swap_keywords_NOT_EQUAL_SYMB,

	swap_keywords_AND,
	swap_keywords_OR,
	swap_int_to_hex_repr,
	swap_int_to_select_repr,

	spaces_to_whitespaces_space_tab,
	spaces_to_whitespaces_space_newline,
	spaces_to_whitespaces_space_f,
	spaces_to_whitespaces_space_vert,
	spaces_to_whitespaces_space_extraspace,
	spaces_to_whitespaces_space_comment_newline,
	multiline_comment_rewriting,
	singleline_comment_rewriting,

	change_num_tautologies_EQUAL,
	change_num_tautologies_NOTEQUAL,
	change_num_tautologies_IN,

	change_string_tautologies_EQUAL1,
	change_string_tautologies_EQUAL2,
	change_string_tautologies_NOTEQUAL1,
	change_string_tautologies_NOTEQUAL2,
	logical_invariant,
	reset_inline_comments
]

# strategies = [ reset_inline_comments, logical_invariant]

place_mutation = [
	"first",
	"random",
	"all"
	]
