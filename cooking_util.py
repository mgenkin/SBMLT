import numpy as np
import json
import matplotlib.pyplot as plt
import sklearn

def load_data():
	# return a list of recipes (dict) with lists of ingredients under 'ingredients' and id's
	# the training set also has data
	train_set = json.load(open("train.json"))
	test_set = json.load(open("test.json"))
	m = len(train_set)

	return train_set, test_set


class Data_mapper():
	# maps data to arrays, stores data mapping in cuisine_dict and ingredients_dict
	def __init__(self):
		self.initialized = False
		self.ingredients_dict = None
		self.cuisine_dict = None
		pass

	def get_dicts(self):
		return self.ingredients_dict, self.cuisines_dict

	def make_train_arrays(self, data):
		# take an array of data in the dictionary-list format and turn it into X and y 
		if not self.initialized:
			ing_ctr = 0
			cuis_ctr = 0
			ingredients_dict = {}
			cuisines_dict = {}
			for recipe in data:
			    for ingredient in recipe['ingredients']:
			        if ingredient not in ingredients_dict:
			            ingredients_dict[ingredient] = ing_ctr
			            ing_ctr += 1
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
				X_train[i, ingredients_dict[ingredient]] = 1

		return X_train, y_train

	def make_test_array(self, data):
		if not self.initialized:
			raise Exception("no ingredient dictionary stored, please map training data first.")

		X_test = np.zeros((len(data),len(self.ingredients_dict)))

		for i in xrange(len(data)):
			recipe = data[i]
			for ingredient in recipe['ingredients']:
				if ingredient in self.ingredients_dict:
					X_test[i, self.ingredients_dict[ingredient]] = 1
				else:
					continue

		return X_test