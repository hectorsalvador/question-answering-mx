## Data

# https://github.com/RaRe-Technologies/gensim/blob/2a70e3a726404cd4230542a35cfd2dc4d63da6f1/gensim/models/wrappers/fasttext.py#L246
# https://rare-technologies.com/fasttext-and-gensim-word-embeddings/

import logging
from gensim.models import KeyedVectors
from gensim.models import word2vec


class W2V_Model(object):

	def __init__(self):
		self.model = None 

	def load(self, modelfile, binary = True):
		self.model = KeyedVectors.load_word2vec_format('Data/' + modelfile, binary= binary)


	def train(self, sentences):
		if not isinstance(self.model, gensim.models.keyedvectors.KeyedVectors):
			self.model = word2vec.Word2Vec(sentences, size=200)
		
		else: 
			print("You have already trained a model, you can't train a new one")
			return 

	def similarity(self, words, top_n = 20):
		
		results = self.model.most_similar(words, topn = top_n)
		results_list = []
		for word, score in results:
			results_list.append(word)

		return results_list

	def find_concepts(self, positive, negative, top_n = 20):
		results = self.model.most_similar(positive = positive,
		negative = negative, topn = top_n)
		results_list = []
		for word, score in results:
			results_list.append(word)

		return results_list

	def intruder(self,words):

		results = self.model.doesnt_match(words)

		return results


if __name__ == '__main__':

	modelfile = 'SBW-vectors-300-min5.txt'
	w2v = W2V_Model()
	w2v.load(modelfile, False)

	print(w2v.model.most_similar(positive=['rey', 'mujer'], negative=['hombre'], topn=1))


