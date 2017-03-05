### Index creation
### Hector Salvador Lopez

from nltk.stem.snowball import SpanishStemmer
import json

########################
### Helper functions ###
########################

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
	words = text.split()
	return words

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
		if stem:
			word = stemmer.stem(word)
		if word not in index:
			index[word] = 0
		index[word] += 1
	return index

def save_index_to_json(index, json_name):
	with open(json_name, 'w', encoding='utf-8') as f:
		json.dump(index, f, ensure_ascii=False, indent=4, sort_keys=True)

########################
#### Main functions ####
########################

### Build word count
def build_word_indices(files, stem):
	'''
	wrapper
	'''
	for prefix in files:
		filename = 'leyes/{}.txt'.format(prefix)
		text = get_text(filename)
		words = preprocess_text_to_words(text)
		index = build_index_from_words(words, stem)
		json_name = 'indices/{}.json'.format(prefix + stem*'_stem')
		save_index_to_json(index, json_name)

### Build inverted index
def build_inverted_index(files, inv_index_name, stem):
	inv_index = {}
	for prefix in files:
		filename = 'indices/{}.json'.format(prefix + stem*'_stem')
		f = open(filename, 'r', encoding='utf-8')
		index = json.load(f)
		for key in index.keys():
			if key not in inv_index:
				inv_index[key] = []
			inv_index[key].append(prefix)
	save_index_to_json(inv_index, inv_index_name)


########################
### 	Wrapper 	 ###
########################

### I should have a wrapper function here maybe?
### I don't know which json will work better, stemmed or not. So let's
### try both for now

leyes = ['LH', 'codigocivil']
stem = True
build_word_indices(leyes, stem)
build_inverted_index(leyes, 'indices/inverted.json', stem)

# sentences = [nltk.pos_tag(sent, lang='spa') for sent in sentences]

# for i in range(len(sentences)):
#     for tup in sentences[i]:
#         if tup[0] == "multa":
#             print(i)
