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

path_pfx = os.getcwd()

def load_inverted_index(stem):
	filename = path_pfx + '/indices/inverted{}.json'.format(stem*'_stem')
	f = open(filename, 'r', encoding='utf-8')
	index = json.load(f)
	return index 

def load_index_p(doc, stem):
	filename = path_pfx + '/indices/{}_p.json'.format(doc + stem*'_stem')
	f = open(filename, 'r', encoding='utf-8')
	index = json.load(f)
	return index 	

def preprocess_words(words, stem):
	stemmer = SpanishStemmer()
	processed = []
	for word in words:
		if stem: 
			word = stemmer.stem(word)
		processed.append(word) 
	return processed

def get_documents(word, inverted_index):
	doc_set = set()
	if word in inverted_index:
		doc_set = doc_set.union(inverted_index[word])
	return doc_set

## http://stevenloria.com/finding-important-words-in-a-document-using-tf-idf/
def tf(word, document):
	'''
	Takes,
		word, a string
		document, a blob object

	Returns term frequency (TF)		
	'''
	return document.words.count(word) / len(document.words)

def n_containing(word, doclist):
    return sum(1 for doc in doclist if word in doc.words)

def idf(word, doclist):
    return math.log(len(doclist) / (1 + n_containing(word, doclist)))

def tfidf(word, document, doclist):
    return tf(word, document) * idf(word, doclist)

def bm25(word, document, doclist, l,  k=2.0, b=0.75):
	return idf(word, doclist) * tf(word, document) * (k + 1) /\
			tf(word, document) + k * (1 - b + b * len(document)/l)
##

def where_should_i_look(words, stem):
	'''
	words is a list of strings, for now:
	['robo', 'algun', 'sentencia']
	'''
	inverted = load_inverted_index(stem)
	word = words[0]
	doc_set = get_documents(word, inverted)

	if len(words) > 1:
		for word in words[1:]:
			docs = get_documents(word, inverted)
			doc_set.intersection(docs)

	docs = list(doc_set)
	print('Working with the following documents:\n{}'.format(docs))

	texts = []

	for prefix in docs:
		filename = path_pfx + '/leyes/{}.txt'.format(prefix)
		text = get_text(filename).split('\n')
		inverted_index = load_index_p(prefix, stem)

		for word in words:
			# right now, this is get_documents, but it could be 
			# anything else
			paragraphs = list(get_documents(word, inverted_index))
			texts.extend([(text[x], prefix) for x in paragraphs])

	return texts

def score_docs(documents, words):
	'''
	documents, a list of strings (paragraphs)
	words, a list of strings (words)
	'''
	rv = []
	blobs = [tb(paragraph) for paragraph in documents]
	l = [list(doc.words) for doc in blobs]
	l = len(set([item for sublist in l for item in sublist]))

	for i, blob in enumerate(blobs):
	    scores = {wd: bm25(wd, blob, blobs, l) for wd in blob.words}
	    score = np.array([scores.get(wd, 0) for wd in words]).sum()
	    rv.append(score)
	return rv

def show_top_docs(k, documents, scores):
	'''
	documents, list of tuples with (text, law)
	scores, a list of floats with tfidf scores
	'''
	data = [[tup[0], tup[1], sc] for tup, sc in zip(documents, scores)]
	df = pd.DataFrame(data=data, columns=['documents', 'law', 'scores'])
	df.sort_values('scores', ascending=False, inplace=True)
	return df.head(k)

def texts(words, k, stem=True):
	'''

	'''
	words = preprocess_words(words, stem)
	if stem:
		print('Working with stemmed words: {}'.format(words))
	else:
		print('Working with words: {}'.format(words))
	
	documents = where_should_i_look(words, stem)
	paragraphs = [tup[0] for tup in documents]
	texts = [' '.join(preprocess_words(paragraph.split(), stem)) for paragraph in paragraphs]
	scores = score_docs(texts, words)
	df = show_top_docs(k, documents, scores)

	return df

# words = ["robo", "hidrocarburos", 'contrabando', "multa", "sanción",\
# 		 "ducto", "pepino", "hurto", "enajenación", "gasolina", "diésel",\
# 		 'sancionará', 'castigará']
words = ["robo", "hidrocarburos", 'contrabando', "multa"]

start = time.time()
texts(words, 3)
print('It took {} seconds.'.format(time.time()-start))



