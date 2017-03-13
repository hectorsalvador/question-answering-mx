### Index creation
### Hector Salvador Lopez

'''
These functions generate word indices, later used to find 
quickly word appearance on documents. Indices are stored as json
files in a folder called indices/.
'''
from nltk.stem.snowball import SpanishStemmer
import json
import os 
import pandas as pd

########################
### Helper functions ###
########################

STOP_WORDS = ['la', 'las', 'el', 'los', 'lo', 'y', 'o', 'que', 'se',\
 'a', 'con', 'de', 'en', 'por', 'para', 'no', 'al', 'del', 'si',\
 'un', 'una', 'este', 'hay', 'como', 'su', 'asi']

path_pfx = os.getcwd()

def get_text(filename):
	'''
	Takes:
	- filename, a string indicating a text file to open
	It assumes text file will be encoded in utf-8 format.

	Returns:
	- text, a string of characters
	'''
	f    = open(filename, 'r', encoding = 'utf-8')
	text = str(f.read())
	return text

def preprocess_text_to_words(text):
	'''
	Takes: 
	- text, a string of characters
	
	Returns:
	- words, a list of strings
	'''
	PUNCTUATION = '".,;:()/-|…“'
	for p in PUNCTUATION:
		text = text.replace(p, '')
	words, paragraphs = text.split(), text.split('\n')
	return words, paragraphs

def build_index_from_words(words, stem):
	'''
	Takes: 
	- words, a list of strings
	
	Returns:
	- index, a dictionary with a count of times a word appears
	in the document
	'''
	index = {}
	stemmer = SpanishStemmer()
	for word in words:
		if word not in STOP_WORDS:
			if stem:
				word = stemmer.stem(word)
			if word not in index:
				index[word] = 0
			index[word] += 1
	return index

def build_inverted_index(index, inv_index, prefix):
	for key in index.keys():
		if key not in inv_index:
			inv_index[key] = []
		inv_index[key].append(prefix)

def track_stop_words(index, stop_words, top=20):
	'''
	Only used this function to find the highest 20 occurring words.
	Then deleted manually the ones that seemed most appropriate, and
	stored them in STOP_WORDS.
	'''
	common_words = sorted(index, key=index.get, reverse=True)[:top]
	new_words = {x: index[x] for x in common_words}
	for key, val in new_words.items():
		if key in stop_words:
			stop_words[key] += new_words[key]
		else:
			stop_words[key] = new_words[key]

def save_index_to_json(index, json_name):
	with open(json_name, 'w', encoding='utf-8') as f:
		json.dump(index, f, ensure_ascii=False, indent=4, sort_keys=True)

########################
#### Main functions ####
########################

### Build word count
def build_word_indices(files, stem):
	'''
	wrapper Do I really need this word count? I think I do, for
	tracking the stop words
	'''
	stop_words = {} #keep track of most common words
	inv_index = {}

	for prefix in files:
		print("Processing {}..".format(prefix))
		filename = path_pfx + '/leyes/{}.txt'.format(prefix)

		#preprocess
		try:
			text = get_text(filename)
		except:
			continue
		words, paragraphs = preprocess_text_to_words(text)
		
		#build word count index, inverted index, and paragraph index
		index = build_index_from_words(words, stem)
		p_index = build_paragraph_inv_index(paragraphs, stem)
		track_stop_words(index, stop_words)
		build_inverted_index(index, inv_index, prefix)
		
		#save and print that done
		index_name = path_pfx + '/indices/{}.json'.format(prefix + stem*'_stem')
		save_index_to_json(index, index_name)

		p_index_name = path_pfx + '/indices/{}_p.json'.format(prefix + stem*'_stem')
		save_index_to_json(p_index, p_index_name)
	
		print("  ..finished with {}.".format(prefix))

	inverted_filename = path_pfx + '/indices/inverted{}.json'.format(stem*'_stem')
	save_index_to_json(inv_index, inverted_filename)

	return stop_words

def build_paragraph_inv_index(paragraphs, stem):
	p_index = {}
	stemmer = SpanishStemmer()
	for i, paragraph in enumerate(paragraphs):
		words = [word for word in paragraph.split() if word not in STOP_WORDS]
		for word in words:
			if stem:
				word = stemmer.stem(word)
			if word not in p_index:
				p_index[word] = []
			p_index[word].append(i)
	return p_index


########################
### 	Wrapper 	 ###
########################

def go(stem=True, show_stop_words=False):
	''' Wrapper function'''
	if stem:
		print("Using stemmed words.")
	else:
		print("Using non-stemmed words.")

	leyes = pd.read_csv('docnames.csv')
	stop_words = build_word_indices(list(leyes.ix[:,0]), stem)

	if show_stop_words:
		return stop_words
	else:
		print("Finished processing documents. Stored in their respective folders.")
