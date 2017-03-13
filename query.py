#Generate query

from question_processing import Question
from synonyms import Synonyms
from Word2VecModel import W2V_Model 
from spanish_tagger import Spanish_Postagger
import json

class Query(object):

	def __init__(self, APIfile, keyfile, tagfile, jarfile, 
		w2vfile):

		self.API = Synonyms(APIfile, keyfile)
		self.W2V = W2V_Model()
		self.W2V.load(w2vfile, False)
		self.Tagger = Spanish_Postagger(tagfile, jarfile)
		self.Tagger.load_tagger()
		self.question = None 
		self.query = []
		self.qtype = None 
	

	def set_question(self, question, question_json):
		self.query = []
		self.question = Question(question, 
		question_json)
		self.question.load_json_questiontype()
		self.question.tag_question(self.Tagger)
		self.question.get_key_words()
		self.question.get_question_type()
		self.qtype = self.question.question_type 


	def remove_stop_words(self, jsonfile):
		
		with open(jsonfile) as json_data:
			stop_words = json.load(json_data)

		stop_words = stop_words["stopwords"]

		for verb in self.question.list_verbs:
			if verb in stop_words:
				self.question.list_verbs.remove(verb)

		self.query = self.question.list_nouns + self.question.list_verbs


	def convert_verbs(self):
		for verb in self.question.list_verbs:
			self.W2V.model.most_similar( positive = [verb], 
				negative = ["comer", "ir", "hacer", "estar", "pasear"], topn = 1)


	def find_vectors (self):
		list_results = []

		for noun in self.question.list_nouns:
			list_results = self.W2V.similarity(noun, top_n = 5)
			list_results += list_results

		for verb in self.question.list_verbs:
			list_results = self.W2V.similarity(verb, top_n = 5) 
			list_results += list_results

		for result in list_results:
			self.query.append(result)


	def find_synonyms(self):
		for noun in self.question.list_nouns:
			list_noun = self.API.get(noun)[0:2]
			self.query += list_noun

		for verb in self.question.list_verbs:
			list_verbs = self.API.get(verb)[0:2]

			self.query += list_verbs

	def get_query(self, stopwords):

		self.remove_stop_words(stopwords)
		self.find_vectors()
		self.find_synonyms()
		self.query = set(self.query)

	def add_words(self, word_list):
		
		self.query.update(word_list)


	def remove_words(self, word_list):
		for word in word_list:
			self.query.remove(word)


if __name__ == '__main__':

		# Data for Tagger
	tagfile = 'stanford-postagger-full-2016-10-31/models/spanish.tagger'
	jarfile = 'stanford-postagger-full-2016-10-31/stanford-postagger.jar'

	# Data for synonyms
	APIfile = 'http://store.apicultur.com/api/sinonimosporpalabra/1.0.0/' 
	keyfile = 'f7JE_2svUVwP5ARGfw8aQhnLXlga'

	# Data for w2v model
	w2vfile = 'SBW-vectors-300-min5.txt'

	# Data for question_type
	question_json = 'Data/question_type.json'
	stopwords = 'Data/stopwords.json'

	# Question

	question = "¿Qué pasa si mato a alguien?"

	
	query = Query(APIfile, keyfile, tagfile, jarfile, 
		w2vfile)

	query.set_question(question, question_json)

	query.remove_stop_words(stopwords)

	query.find_vectors()

	query.find_synonyms()

	print(query.query)
	print(query.qtype)








