import json
import numpy as np
from pprint import pprint
import sklearn

with open("trane.json") as data_file:
    data = json.load(data_file)

ingredientsArray = []
cuisinesArray = []
    
for recipe in data:
    for ingredient in recipe['ingredients']:
        if ingredient not in ingredientsArray:
            ingredientsArray.append(ingredient)
    if recipe['cuisine'] not in cuisinesArray:
        cuisinesArray.append(recipe['cuisine'])
        
ingredientsMatrix = np.zeros(shape=(len(cuisinesArray),len(ingredientsArray)))

for recipe in data:
    for ingredient in recipe['ingredients']:
        ingredientsMatrix[cuisinesArray.index(recipe['cuisine'])][ingredientsArray.index(ingredient)] = 1
            
pprint(ingredientsMatrix)
cam=np.matrix(cuisinesArray)
cam.transpose()
#def eval_measure(y1, y2):
#    return (y1 == y2).mean()
#split = 0.8*X.shape[0]
#clf = tree.DecisionTreeClassifier()
#clf.fit(X[:split], Y[:split])
#y_pred = clf.predict(X[split:])
#eval_measure(y_pred, Y[split:])
