from nltk.tag import StanfordPOSTagger

class Spanish_Postagger(object):
	"""docstring for Spanish_Postagger"""
	
	def __init__(self, model, jar ):		
		self.model = model
		self.jar = jar
		self.tagger_model = None 

	def load_tagger(self):

		self.tagger_model = StanfordPOSTagger( model_filename = self.model,
			path_to_jar = self.jar, encoding = 'utf8')

	def tag(self, word_list):
		list_results = self.tagger_model.tag(word_list)
		return list_results