#Create an API 

# https://store.apicultur.io:9443/apis/info?name=SinonimosPorPalabra&version=1.0.0&provider=molinodeideas
# http://stackoverflow.com/questions/29931671/making-an-api-call-in-python-with-an-api-that-requires-a-bearer-token

#curl -H "Authorization: Bearer f7JE_2svUVwP5ARGfw8aQhnLXlga" 'http://store.apicultur.com/api/sinonimosporpalabra/1.0.0/beso'

import requests
import sys

class Synonyms(object):
	
	def __init__(self, api, authorization):
		self.api = api
		self.authorization = authorization


	def get(self, word):
		query = self.api + word
		print(query)
		token = 'Bearer ' + self.authorization 
		print(token)
		resp = requests.get(query, 
		headers={'Authorization': token})

		try:
			json_list = resp.json()
			results = []
			for i in json_list:
				value = i['valor']
				results.append(value) 

		except:
			results = []

		return results
		
		
if __name__ == '__main__':

	API = 'http://store.apicultur.com/api/sinonimosporpalabra/1.0.0/' 
	KEY = 'f7JE_2svUVwP5ARGfw8aQhnLXlga'

	c = Synonyms(API, KEY)
	result_1 = query.get("beso")
	print(result_1)
	result_2 = query.get("comida")
	print(result_2)



