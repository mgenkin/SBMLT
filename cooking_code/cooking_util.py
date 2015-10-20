import numpy as np
import json
import matplotlib.pyplot as plt
import sklearn
import collections
import string
import re
import csv

def load_data():
	# return a list of recipes (dict) with lists of ingredients under 'ingredients' and id's
	# the training set also has data
	train_set = json.load(open("train.json"))
	test_set = json.load(open("test.json"))
	m = len(train_set)

	return train_set, test_set

def strclean(str_in):
	str_in = re.sub(r'[^\sa-zA-Z]', '', str_in).lower().strip() #trim non-letters, spaces, and make it lowercase
	if len(str_in)<=3: #too short to care about
		return str_in
	#the rest of this is to fix plurals
	if str_in[-1] != 's': # not a plural, so we don't care
		return str_in
	str_in = str_in.rstrip('s') # remove 's' or 'es', change ending 'i' to 'y'
	if str_in[-1] != 'e':
		return str_in
	if str_in[-1] == 'e' and str_in[-2]!='i':
		return str_in
	str_in = str_in.rstrip('e')
	str_in = str_in.rstrip('i')
	str_in = str_in+'y'
	return str_in

def CleanData(train,test):
	# Cleaning all the data and collecting one word ingredients
	cleaned = []
	singles = []
	#to see an example of how it works,  uncomment the print statements
	#print data[0]["ingredients"]
	for recipe in train:
		for item in recipe['ingredients']:
			item = strclean(item)
			if ' ' not in item:
				singles.append(item)
			cleaned.append(item)
		recipe.update({'ingredients':cleaned})
		cleaned=[]
	
	#sorting by frequency and removing repititions
	count = collections.Counter(singles)
	singles = sorted(count,key=count.get,reverse=True)
	
	#mapping all multi-word ingredients to one word ingredients
	for recipe in train:
		for item in recipe['ingredients']:
			if ' ' in item:
				words = item.split()
				frequency=len(singles)-1
				for word in words:
					if word in singles:
						if frequency>singles.index(word):
							frequency=singles.index(word)
				item = singles[frequency]
			cleaned.append(item)
		recipe.update({'ingredients':cleaned})
		cleaned = []

	for recipe in test:
		for item in recipe['ingredients']:
			if ' ' in item:
				words = item.split()
				frequency=len(singles)-1
				for word in words:
					if word in singles:
						if frequency>singles.index(word):
							frequency=singles.index(word)
				item = singles[frequency]
			cleaned.append(item)
		recipe.update({'ingredients':cleaned})
		cleaned = []
	#print data[0]["ingredients"]
	return train,test

def clean_data_bow(train, test, min_recipes=50):
	# segments words in ingredients into separate ingredients
	# for example "wheat bread" becomes 2 ingredients: "wheat" and "bread"
	# considers only words which appear in more than min_recipes recipes
	train_bow, test_bow, all_words = [], [], {}
	
	# all_words stores word frequency
	# train_bow and test_bow will be output
	for recipe in train:
		for item in recipe['ingredients']:
			words_in_item = [strclean(w) for w in item.split()] # split into words and clean each word
			for word in words_in_item:
				if word in all_words:
					all_words[word]+=1 # add to word frequency
				else:
					all_words[word]=1 # occurs in one recipe

	for word in all_words.keys(): # remove rare words
		if all_words[word] <= min_recipes:
			all_words.pop(word)

	# 
	for recipe in train:
		cleaned = []
		for item in recipe['ingredients']:
			words = [strclean(w) for w in item.split()]
			for word in words:
				if word in all_words.keys():
					cleaned.append(word)
		recipe.update({'ingredients':cleaned})

	for recipe in test:
		cleaned = []
		for item in recipe['ingredients']:
			words = [strclean(w) for w in item.split()]
			for word in words:
				if word in all_words.keys():
					cleaned.append(word)
		recipe.update({'ingredients':cleaned})

	return train, test
	
class Data_mapper():
	# maps data to arrays, stores data mapping in cuisine_dict and ingredients_dict
	# ingredients_dict maps ingredient to its column number
	# cuisine_dict maps cuisine to a number
	def __init__(self):
		self.initialized = False # indicates whether ingredients_dict, cuisine_dict exist
		self.ingredients_dict = None
		self.cuisine_dict = None

	def get_dicts(self):
		return self.ingredients_dict, self.cuisine_dict

	def make_train_arrays(self, data):
		# take an array of data in the dictionary-list format and turn it into X and y 
		if not self.initialized: # no stored dictionaries yet, so we need to make them
			ing_ctr = 0
			cuis_ctr = 0
			ingredients_dict = {}
			cuisines_dict = {}
			for recipe in data:
			    for ingredient in recipe['ingredients']:
			        if ingredient not in ingredients_dict: # haven't seen this one yet
			            ingredients_dict[ingredient] = ing_ctr # add it to the dictionary
			            ing_ctr += 1 # increment counter for next ingredient
			    if recipe['cuisine'] not in cuisines_dict:
			        cuisines_dict[recipe['cuisine']] = cuis_ctr
			        cuis_ctr += 1
			self.ingredients_dict = ingredients_dict
			self.cuisine_dict = cuisines_dict
			self.initialized = True

		X_train = np.zeros((len(data),len(self.ingredients_dict)))
		y_train = np.zeros((len(data)))

		for i in xrange(len(data)):
			recipe = data[i]
			cuisine = recipe['cuisine']
			y_train[i] = cuisines_dict[cuisine]
			for ingredient in recipe['ingredients']: 
				X_train[i, ingredients_dict[ingredient]] = 1 # place a 1 in the appropriate slot

		return X_train, y_train

	def make_test_vector(self, recipe):
		# maps a single recipe to its vector representation
		
		if not self.initialized:
			raise Exception("no ingredient dictionary stored, please map training data first.")

		X_test = np.zeros(len(self.ingredients_dict))
		for ingredient in recipe['ingredients']:
			if ingredient in self.ingredients_dict:
				X_test[self.ingredients_dict[ingredient]] = 1
			else:
				continue
		return X_test

def submit(classifier, test_set, dm, filename='sub.csv'):
	# takes your classifier and makes a kaggle-style submission csv file out of it
	# Your submission function should take an element of your test set and return the predicted cuisine (string).
	# Pass in your datamapper instance as dm
	with open(filename, 'wb') as csvfile:
		w = csv.DictWriter(csvfile, ["id", "cuisine"])
		w.writeheader()
		_, cuis_dict = dm.get_dicts()
		cuis_dict_rev = {v: k for k, v in cuis_dict.items()}
		for recipe in test_set:
			recipe_vector = dm.make_test_vector(recipe)
			pred = classifier(recipe_vector)[0]
			w.writerow({"id":recipe["id"], "cuisine":cuis_dict_rev[pred]})