def NN_Node(object):
	def __init__(self, input_shape, output_shape):
		self.W 
		self.b

	def get_output(self, input):

		return output

	def update_gradient(self, cost):
		# cost may be a list
		# update W and b to new values


def NodeLayer(object):
	def __init__(self, input_shape, output_shape, num_nodes):
		self.nodes = # a list of nodes, initialized with the proper shapes

	def get_output(self, input):
		# compute the output of the layer
		return output

	def compute_gradient(self, cost):
		# cost may be a list of costs from the neurons of the next layer


def NeuralNetwork(object):
	
	def __init__(self, in_size, layer_sizes, out_size):

	def feedforward_output(self, input):

		return output

	def update_gradient(self, cost):
		# implement backpropagation through the layers



nn = NeuralNetwork(10, [100], 10)
for i in range(500):
	y_pred = nn.feedforward_output(data)
	cost = (y_true - y_pred)**2
	nn.update_gradient(cost)