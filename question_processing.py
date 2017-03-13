## Question Processing code
## Carlos Grandet and Hector Salvador

# This code takes a question and process it into a query to feed
# our question answering machine 

import os
from nltk.tokenize import RegexpTokenizer
import json 


class Question(object):
	"""docstring for ClassName"""
	
	def __init__(self, question, json_type):
		
		self.question = question
		self.tagged_question = None 
		self.list_nouns = []
		self.list_verbs = []
		self.list_interrogatives = []
		self.json_qtype = json_type
		self.probability_type = {}
		self.question_type = [] 

	def tag_question(self, post_tagger):

		tokenizer = RegexpTokenizer(r'\w+')
		question_tokens = tokenizer.tokenize(self.question)
		self.tagged_question = post_tagger.tag(question_tokens)

	def get_key_words(self):
		
		list_nouns = []
		list_verbs = []
		list_interrogatives = []
		
		for word, tag in self.tagged_question:
			if tag[0] == "n":
				self.list_nouns.append(word)
			if tag[0] == "v":
				self.list_verbs.append(word)
			if tag == "dt0000" or tag == "pt000000" or tag == "pr000000":
				self.list_interrogatives.append(word)


	def load_json_questiontype(self):
		with open(self.json_qtype) as json_data:
			self.dictionary_qtype = json.load(json_data)
   

	def get_question_type(self):

		self.probability_type = {} 
		for qtype, value_list in self.dictionary_qtype.items():
			size = len(value_list)
			self.probability_type[qtype]  = 0
			for value in value_list:
				if value in self.list_interrogatives or value in self.list_verbs or value in self.list_nouns:
					
					probability = 1 / size
					old_probability = self.probability_type.get(qtype, 1)
					if old_probability == 0:
						old_probability = 1
					updated_probability = old_probability*probability
					self.probability_type[qtype] = updated_probability

			for noun in self.list_nouns:
				if noun in value_list:
					self.list_nouns.remove(noun)

			for verb in self.list_verbs:
				if verb in value_list:
					self.list_verbs.remove(verb)


		best_prob = 0
		best_type = None

		self.question_type = []
		for qtype, prob in self.probability_type.items():
			if prob >= best_prob: 
				best_type = qtype
				best_prob = prob

		self.question_type.append(best_type)

		for qtype, prob in self.probability_type.items():
			if abs(best_prob - prob) < best_prob*.85 and qtype not in self.question_type:
				self.question_type.append(qtype)

		if self.question_type == [None]:
			self.question_type = list(self.probability_type.keys)














	


















