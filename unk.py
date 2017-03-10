### Querying
### Hector Salvador Lopez

# This file helps to retrieve documents where we should be 
# looking for occurrences of a list of words. 

from nltk.stem.snowball import SpanishStemmer
from index import get_text, preprocess_text_to_words
import json
import os

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

	paragraphs = []

	for prefix in docs:
		filename = path_pfx + '/leyes/{}.txt'.format(prefix)
		text = get_text(filename).split('\n')
		inverted_index = load_index_p(prefix, stem)

		texts = []
		for word in words:
			# right now, this is get_documents, but it could be 
			# anything else
			paragraphs = list(get_documents(word, inverted_index))
			texts.extend([text[x] for x in paragraphs])

	return texts

def texts(words, stem=True):
	words = preprocess_words(words, stem)
	if stem:
		print('Working with stemmed words: {}'.format(words))
	else:
		print('Working with words: {}'.format(words))
	return where_should_i_look(words, stem)



