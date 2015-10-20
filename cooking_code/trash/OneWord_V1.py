#!/usr/bin/python

import json
import string
import numpy as np
import re

def strclean(str_in):
	if len(str_in)<4:
		return str_in
	str_in = str_in.lower()
	str_in = re.sub(r'[^\sa-zA-Z]', '', str_in).lower().strip()
	if str_in[-1] != 's':
		return str_in
	str_in = str_in.rstrip('s')
	if str_in[-1] != 'e':
		return str_in
	if str_in[-1] == 'e' and str_in[-2]!='i':
		return str_in
	str_in = str_in.rstrip('e')
	str_in = str_in.rstrip('i')
	str_in = str_in+'y'
	return str_in

names = []
cleaned = []
count = 0
unique = []

with open("train.json") as json_file:
	raw = json.load(json_file)
print 'Loaded Data'
for recipe in raw:
	cuis = recipe['cuisine']
	if cuis not in names:
		names.append(cuis)
		cleaned.append([])
	if cuis in names:
		i=names.index(cuis)
	ingred = recipe['ingredients']
	for item in ingred:
		# words=item.split()
		item = strclean(item)
		if ' ' not in item and item not in cleaned[i]:
			cleaned[i].append(item)
		if ' ' not in item and item not in unique:
			unique.append(item)


print 'List of all unique one word ingredients by culture:'
total = 0
for cuisine in names:
	print cuisine,len(cleaned[names.index(cuisine)])
	total = total+len(cleaned[names.index(cuisine)])
print 'There are ',len(unique),' unique one word ingredients'
