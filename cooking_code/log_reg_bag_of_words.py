import numpy as np
import cooking_util
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

train_set, test_set = cooking_util.load_data()
train_set_c, test_set_c = cooking_util.clean_data_bow(train_set, test_set, min_recipes=50)
dm = cooking_util.Data_mapper()
X_train, y_train = dm.make_train_arrays(train_set_c)

clf = LogisticRegression()
fold_scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
mean_fold_score = fold_scores.mean()
mean_fold_std = fold_scores.std()
print("Cross-val accuracy: %0.5f (+/- %0.5f)" % (mean_fold_score, mean_fold_std *2))

clf.fit(X_train, y_train)

print "writing submission file to sub_log_bow.csv"
cooking_util.submit(clf.predict, test_set_c, dm, filename='sub_log_bow.csv')
print "finished"