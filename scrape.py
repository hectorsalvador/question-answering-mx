### Web scraping script
### Hector Salvador
'''
This script retrieves the txt files from the following website:
http://www.diputados.gob.mx/LeyesBiblio/index.htm
It also saves the names of the laws for future reference. They're
saved in a file called "docnames.csv"
Scraper working as of 3/13/2017.
''' 

import bs4
import re
import requests
import csv
import os

URL = 'http://www.diputados.gob.mx/LeyesBiblio/index.htm'
PREFIX = "http://www.diputados.gob.mx/LeyesBiblio/"

def go():
	r = requests.get(URL)
	soup = bs4.BeautifulSoup(r.text, 'html5lib')
	tags = soup.find_all('a')
	name, docs = [], []

	for tag in tags:
		str_tag = str(tag).replace('"', "'")
		title_tag = re.findall(r"href=\'ref\/[A-Za-z0-9.-_]*\'", str_tag)
		if title_tag:
			name.append(tag.text.strip().replace('\n', '').replace('\t', ''))
		urls = re.findall(r"href=\'doc\/[A-Za-z0-9.-_]*\'", str_tag)

		for result in urls:
			ref = result.strip('href=').strip("'")
			docs.append(ref.strip('doc/.'))
			path = os.getcwd() + '/' + ref
			if os.path.isfile(path):
				print('{} is already on disk.'.format(ref))
				continue
			url = PREFIX + ref
			doc_req = requests.get(url)
			print("Fetching {}".format(url))
			with open(ref, 'wb') as f:
				f.write(doc_req.content)

	with open('docnames.csv', 'w') as f:
		writer = csv.writer(f)
		#writer.writerow(['id', 'ley'])
		name.pop(0) # extra tag at the beginning "Reformas a la Constitución"
		writer.writerows(zip(docs, name))
