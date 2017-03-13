### Querying
### Hector Salvador Lopez

# This file helps to retrieve documents where we should be 
# looking for occurrences of a list of words. 

from nltk.stem.snowball import SpanishStemmer
from index import get_text, preprocess_text_to_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import json
import os
import math
from textblob import TextBlob as tb
import time
from jellyfish import jaro_winkler
from ngrams import retrieve_model


class ScoreParagraphs(object):

	def __init__(self, question, words, stem):
		self.question = question
		self.stem = stem
		self.stemmer = SpanishStemmer()
		self.words = words
		self.stemmed_words = self.stem_words(self.words)
		self.path_pfx = os.getcwd()

		self.inverted_index = self.load_doc_inverted_index()
		self.doc_names = self.init_doc_names()
		self.paragraph_indices = {}
		self.paragraph_inverted_indices = {}
		self.results = pd.DataFrame(columns=['text', 'law', 'score'])
		self.load_paragraph_indices()

		self.L = 23055.676666666666
		self.scores = {'tf': {}, 'idf':{}, 'tfidf':{},'n_containing':{},\
					   'score':{}}

	def stem_words(self, words):
		#print('Stemming {}'.format(words))
		processed = []
		if self.stem:
			for word in words:
				word = self.stemmer.stem(word)
				processed.append(word) 
		return processed

	def load(self, filename):
		#print('Trying to load {}'.format(filename))
		f = open(filename, 'r', encoding='utf-8')
		index = json.load(f)
		#print('Success!')
		return index

	def load_doc_inverted_index(self):
		filename = self.path_pfx + '/indices/inverted{}.json'.format(self.stem*'_stem')
		return self.load(filename) 

	def init_doc_names(self):
		temp = [self.inverted_index[word] for word in self.stemmed_words \
				if word in self.inverted_index]
		return [i for sublist in temp for i in sublist]
		# rv = set()
		# for i, word in enumerate(self.stemmed_words):
		# 	temp = set(self.inverted_index.get(word, []))
		# 	if i == 0:
		# 		rv = rv.union(temp)
		# 	rv = rv.intersection(temp)
		# return list(rv)
	
	def load_paragraph_indices(self):
		for doc in self.doc_names:
			filename = self.path_pfx + '/indices/{}.json'.format(doc + self.stem*'_stem')
			self.paragraph_indices[doc] = self.load(filename) 
			filename = self.path_pfx + '/indices/{}_p.json'.format(doc + self.stem*'_stem')
			self.paragraph_inverted_indices[doc] = self.load(filename)  	
	
	def where_should_i_look(self):
		'''
		words is a list of strings, for now:
		['robo', 'algun', 'sentencia']
		'''
		#print('Working with the following documents:\n{}'.format(self.doc_names))

		for law in self.doc_names:
			filename = self.path_pfx + '/leyes/{}.txt'.format(law)
			paragraph_list = get_text(filename).split('\n')

			for word in self.stemmed_words:
				#print(word)
				paragraphs = self.paragraph_inverted_indices[law].get(word, [])
				#print(paragraphs)
				results = [[paragraph_list[x], law, 0] for x in paragraphs]
				#print(results)
				df_temp = pd.DataFrame(results, columns=['text', 'law', 'score'])
				self.results = self.results.append(df_temp, ignore_index=True, )

		return self.results

	## http://stevenloria.com/finding-important-words-in-a-document-using-tf-idf/
	def tf(self, word, document):
		s = (word, document)
		#print(s, type(s))  
		if s not in self.scores['tf']:
			self.scores['tf'][s] = document.words.count(word) / len(document.words)
		return self.scores['tf'][s]

	def n_containing(self, word, doclist):
		s = (word)
		#print(s, type(s))  
		if s not in self.scores['n_containing']:
			self.scores['n_containing'][s] = sum(1 for doc in doclist if word in doc.words)
		return self.scores['n_containing'][s]

	def idf(self, word, doclist):
		s = (word)
		#print(s, type(s))  
		if s not in self.scores['idf']:
			self.scores['idf'][s] = math.log(len(doclist) / (1 + self.n_containing(word, doclist)))
		return self.scores['idf'][s]

	def tfidf(self, word, document, doclist):
		s = (word, document)
		#print(s, type(s))  
		if s not in self.scores['tfidf']:
			self.scores['tfidf'][s] = self.tf(word, document) * self.idf(word, doclist)
		return self.scores['tfidf'][s]

	def bm25(self, word, document, doclist,  k=2.0, b=0.75):
		'''
		Takes,
			word, a string
			document, a blob object
			doclist, a list with blob objects
			l, 
		Returns term frequency (TF)		
		'''
		s = (word, document)
		#print(s, type(s))  
		if s not in self.scores['score']:
			self.scores['score'][s] = self.idf(word, doclist) * \
			self.tf(word, document) * (k + 1) / (self.tf(word, document) +\
			k * (1 - b + b * len(document)/self.L))
		return self.scores['score'][s]

	def score_docs(self, documents, words, method, k, b):
		'''
		documents, a list of strings (paragraphs)
		words, a list of strings (words)
		'''
		blobs = [tb(paragraph) for paragraph in documents]
		rv = [0]*len(blobs)

		for word in words:
			for i, blob in enumerate(blobs):
				if method == 'word_count':
					rv[i] += blob.words.count(word)
				elif method == 'bm25':
					rv[i] += self.bm25(word, blob, blobs, k, b)
				elif method == 'proximity':
					rv[i] += jaro_winkler(self.question, blob.string)
		self.results['score'] = rv
		return rv

	def drop_duplicates_and_short_paragraphs(self, min_size):
		self.results.drop_duplicates(inplace=True)
		k = lambda x: len(x.split()) > min_size
		self.results = self.results[self.results.text.apply(k)]
		clf = retrieve_model()
		clf.predict(self.results)

	def load_law_names(self, filename):
		rv = pd.read_csv(filename, header=None, names=['law','Law'])
		self.results = self.results.merge(rv, on='law')
		del self.results['law']

	def texts(self, top_k, method, k=2, b=0.75):
		'''

		'''
		words = self.stemmed_words
		if self.stem:
			print('Working with stemmed words: {}'.format(words))
		else:
			print('Working with words: {}'.format(words))
		
		self.where_should_i_look()
		self.drop_duplicates_and_short_paragraphs(4)
		paragraphs = self.results.text
		texts = [' '.join(self.stem_words(paragraph.split())) for paragraph in paragraphs]
		self.score_docs(texts, words, method, k, b)
		self.results.sort_values('score', ascending=False, inplace=True)

		law_filename=self.path_pfx + '/docnames.csv'
		df_names = self.load_law_names(law_filename)

		print(self.results.head(top_k))

		return self.results.head(top_k)

question = 'cuál es la pena por asesinar a mi hermano'
CASTIGOS = ["castigo", "sanción", "multa"]
words = ["asesinar", "persona"] + CASTIGOS
words2 = ["asesino", "asesinar", "homicidio", "matar", "persona",\
		  "individuo", "hermano"]
# words2 = ["robar", "robo", "enajenación", "enajenar", "usurpar", \
# 		"ílicito", "auto", "automóvil", "carro"] + CASTIGOS

# df = pd.DataFrame(columns=[range(5), 'k', 'b'])
# for k in range(0, 80, 5):
# 	for b in range(0, 400, 25):
# 		print(k, b)
test = ScoreParagraphs(question, words2, stem=True)
start = time.time()
#res = test.texts(10, method='proximity', k=k/10, b=b/100)
res = test.texts(20, method='bm25')
#test.texts(words2, 3)
print('It took {} seconds.'.format(time.time()-start))
		# rv = list(res.text).extend([k, b])
		# df = df.append(rv)







