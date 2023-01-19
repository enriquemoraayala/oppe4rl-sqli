from gym_waf.envs.controls import sqlfuzzer as manipulate


#testing the actions/mutations 
if __name__ == '__main__':

	print("\nAction: singleline_comment_rewriting")
	payload = "uni/*1234*/on sel/**/ect-- 1234"
	print("original: ", payload)
	newText = manipulate.singleline_comment_rewriting(payload)
	print("mutation: ",newText)

	print("\nAction: multiline_comment_rewriting")
	payload = "uni/*1234*/on sel/**/ect"
	print("original: ", payload)
	newText = manipulate.multiline_comment_rewriting(payload)
	print("mutation: ",newText)

	print("\nAction: spaces_to_comments")
	payload = "union select"
	print("original: ", payload)
	newText = manipulate.spaces_to_comments(payload)
	print("mutation: ",newText)

	print("\nAction: comments_to_spaces")
	payload = "union/**/select"
	print("original: ", payload)
	newText = manipulate.comments_to_spaces(payload)
	print("mutation: ",newText)	

	print("\nAction: swap_int_to_hex_repr")
	payload = "select 15 "
	print("original: ", payload)
	newText = manipulate.swap_int_to_hex_repr(payload)
	print("mutation: ",newText)	

	print("\nAction: swap_int_to_select_repr")
	payload = "select 15 "
	print("original: ", payload)
	newText = manipulate.swap_int_to_select_repr(payload)
	print("mutation: ",newText)	

	print("\nAction: reset_inline_comments")
	payload = "select 15 /*erferfre*/"
	print("original: ", payload)
	newText = manipulate.reset_inline_comments(payload)
	print("mutation: ",newText)	

	print("\nAction: swap_keywords_NOTEQUAL_NOTLIKE")
	payload = "select * users where 1 <> 4/*erferfre*/"
	print("original: ", payload)
	newText = manipulate.swap_keywords_NOTEQUAL_NOTLIKE(payload)
	print("mutation: ",newText)		

	print("\nAction: change_num_tautologies")
	payload = "select * users where 5 = 5 /*erferfre*/"
	print("original: ", payload)
	newText = manipulate.change_num_tautologies_EQUAL(payload)
	print("mutation: ",newText)	

	print("\nAction: swap_keywords_OR")
	payload = "'  )  )   or   (  (  'x'  )  )   =   (  (  'x"
	print("original: ", payload)
	newText = manipulate.swap_keywords_OR(payload)
	print("mutation: ",newText)	


	print("\nAction: spaces_to_whitespaces_space_comment_newline")
	payload = "'  )  )   or   (  (  'x'  )  )   =   (  (  'x"
	print("original: ", payload)
	newText = manipulate.spaces_to_whitespaces_space_comment_newline(payload)
	print("mutation: ",newText)						

