import numpy as np
import cooking_util
import neural_network as nn
from sklearn.preprocessing import OneHotEncoder
import random
import time
import pickle

train_set, test_set = cooking_util.load_data()
train_set_c, test_set_c = cooking_util.clean_data_bow(train_set, test_set, min_recipes=50)
dm = cooking_util.Data_mapper()
X_train, y_train = dm.make_train_arrays(train_set_c)

y_train = OneHotEncoder(sparse=False).fit_transform(y_train[:,np.newaxis])

learning_rate = 0.1
net = nn.NeuralNetwork([X_train.shape[1], 1000, y_train.shape[1]])
costs = []


print "Starting training"
for j in range(100):
	ind = range(X_train.shape[0])
	random.shuffle(ind)
	start_time = time.time()
	for i in ind:
		y_pred = net.feedforward_output(X_train[i])
		net.backpropagate_gradient(y_train, y_pred, learning_rate)
		cost = ((y_train-y_pred)**2).sum()
		costs.append(cost)
	print "iteration {} took {:.3f}".fit_transformat(j+1, time.time()-start_time)
	print np.sum(costs)
	pickle.dump(net, open('net.pkl', 'w'))
	costs = []

plt.plot(costs)
plt.savefig('costs.png')

# print "writing submission file to sub_rf_bow_mult.csv"
# cooking_util.submit(forest.predict, test_set_c, dm, filename='sub_rf_bow_mult.csv')
# print "finished"