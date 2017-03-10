## Question-answering machine for Mexican federal laws 
## Authors: Hector Salvador Lopez and Carlos Grandet

### Helper functions ###

def read_index():
	'''
	Loads an index with key, value pairs where keys are "key" words and values
	are the documents that contain such word
	'''
	doc_index, pas_index = {}, {}
	return doc_index, pass_index



#### Main functions ####

def query_formulation(question):
	'''
	Translates the text of a question into a structured query that retrieves 
	information	from a document dataset.
	'''
	query = ''
	return query

def answer_detection(question):
	'''
	Based on the text of a question, detects the type of answer that the user
	expects to receive. The type of answer can be any of:
		-
		-
		-
	'''
	ans_type = ''
	return ans_type

def document_retrieval(query, doc_index):
	'''
	Based on a query and an index, retrieves the documents with the highest 
	relevance given the query
	'''
	documents = []
	return documents

def passage_retrieval(query, pass_index, documents):
	'''
	Based on a query and an index, retrieves the documents with the highest 
	relevance given the query
	'''
	passages = []
	return passages

def answer_processing(passages, ans_type):
	'''
	Given selected passages and the type of answer that needs to be returned,
	build the final answet that will be returned to the user.
	'''
	answer_string = ''
	return answer_string


####### Wrapper ########

def get_answers(question):
	doc_index, pass_index = read_index()
	query = query_formulation(question)
	documents = document_retrieval(query, doc_index)
	passages = passage_retrieval(query, pass_index, documents)
	ans_type = answer_detection(question)
	answers = answer_processing(passages, ans_type)
	return answers

# Amitabh mentioned to take a look at the nltk library to see if they are
# doing any of these things alrady. We could take advantage of this.
# Google cloud -> Spark, "Understanding Spark" good book
