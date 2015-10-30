import numpy as np
import cooking_util
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

train_set, test_set = cooking_util.load_data()
train_set_c, test_set_c = cooking_util.clean_data_bow_mult(train_set, test_set, min_recipes=50)
dm = cooking_util.Data_mapper()
X_train, y_train = dm.make_train_arrays(train_set_c)

forest = RandomForestClassifier(n_estimators=100) 
fold_scores = cross_validation.cross_val_score(forest, X_train, y_train, cv=5)
mean_fold_score = fold_scores.mean()
mean_fold_std = fold_scores.std()
print("Cross-val accuracy: %0.5f (+/- %0.5f)" % (mean_fold_score, mean_fold_std *2))

forest.fit(X_train, y_train)

print "writing submission file to sub_rf_bow_mult.csv"
cooking_util.submit(forest.predict, test_set_c, dm, filename='sub_rf_bow_mult.csv')
print "finished"