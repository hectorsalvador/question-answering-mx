## Web scraping script
## Hector Salvador
'''
This script retrieves the txt files from the following website:
http://www.diputados.gob.mx/LeyesBiblio/index.htm
''' 

import bs4
import re
import requests


URL = 'http://www.diputados.gob.mx/LeyesBiblio/index.htm'
PREFIX = "http://www.diputados.gob.mx/LeyesBiblio/"
r = requests.get(URL)
soup = bs4.BeautifulSoup(r.text, 'html5lib')
tags = soup.find_all('a')

documents = []

for tag in tags:
	str_tag = str(tag).replace('"', "'")
	urls = re.findall(r"href=\'doc\/[A-Za-z0-9.-_]*\'", str_tag)
	for result in urls:
		ref = result.strip('href=').strip("'")
		url = PREFIX + ref
		doc_req = requests.get(url)
		print("Fetching {}".format(url))
		#documents.append(url)
		with open(ref, 'wb') as f:
			f.write(doc_req.content)
			