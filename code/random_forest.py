import numpy as np
import cooking_util
import csv # to write the submission files
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

train_set, test_set = cooking_util.load_data()
train_set_c, test_set_c = cooking_util.CleanData(train_set, test_set)
dm = cooking_util.Data_mapper()
X_train, y_train = dm.make_train_arrays(train_set_c)
print X_train, y_train

forest = RandomForestClassifier(n_estimators=200) #initializes the classifier
#cross validation
fold_scores = cross_validation.cross_val_score(forest, X_train, y_train, cv=5)
mean_fold_score = fold_scores.mean()
mean_fold_std = fold_scores.std()
print("Accuracy: %0.5f (+/- %0.5f)" % (mean_fold_score, mean_fold_std *2))

forest.fit(X_train, y_train) # trains the model



with open('sub_randomforest.csv', 'wb') as csvfile: 
    w = csv.DictWriter(csvfile, ["id", "cuisine"])
    w.writeheader()
    _, cuis_dict = dm.get_dicts()
    cuis_dict_rev = {v: k for k, v in cuis_dict.items()}
    for recipe in test_set:
        recipe_vector = dm.make_test_vector(recipe)
        pred_cuis = int(forest.predict(recipe_vector)[0])
        w.writerow({"id":recipe["id"], "cuisine":cuis_dict_rev[pred_cuis]})