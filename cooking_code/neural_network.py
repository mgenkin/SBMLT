import numpy as np


class NN_Node(object):

    def __init__(self, input_shape):
        self.W = np.random.random(input_shape)
        self.b = np.random.random()

    def sigmoid(self, x):
        return 1.0/(1.0+np.exp(-x))

    def get_output(self, node_input):
    	net = self.W.dot(node_input.transpose())+self.b
        output = self.sigmoid(net)
        return output

    def update_params(self, d_cost, node_input, learning_rate):
        # update W and b to new values
        output = self.get_output(node_input)
        sigmoid_prime = output*(1-output)
        grad = d_cost*sigmoid_prime*self.W
        self.W = self.W - learning_rate*grad
        return sigmoid_prime*d_cost

class NodeLayer(object):

    def __init__(self, input_shape, num_nodes):
        # a list of nodes, initialized with the proper shapes
        self.nodes = [NN_Node(input_shape) for i in range(num_nodes)]

    def get_output(self, layer_input):
        # compute the output of the layer
        self.layer_input = layer_input
        output = np.array([node.get_output(layer_input) for node in self.nodes])
        return output

    def get_weight_matrix(self):
        # matrix should be input_shape by num_nodes
        return np.array([node.W for node in self.nodes]).transpose()

    def update_params(self, matrix, learning_rate):
        # matrix has dimension this_layer_nodes by next_layer_nodes
        input_shape = len(self.nodes[0].W)
        # first, update the nodes in this layer
        passing_matrix = np.zeros((input_shape, len(self.nodes)))
        for i in range(len(self.nodes)):
            pass_to_node = np.sum(matrix[i,:])
            d_cost_d_node = self.nodes[i].update_params(pass_to_node, self.layer_input, learning_rate)
            passing_matrix[:,i] = d_cost_d_node
        # then, return d_cost for the previous layer    
        return np.multiply(self.get_weight_matrix(),passing_matrix)


class NeuralNetwork(object):

    def __init__(self, layer_sizes):
        self.layers = [NodeLayer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]

    def feedforward_output(self, net_input):
        data = net_input
        for layer in self.layers:
            data = layer.get_output(data)
        return data

    def backpropagate_gradient(self, y_true, y_pred, learning_rate):
        # implement backpropagation through the layers
        dcost = (y_true - y_pred)
        matrix = np.array(dcost).transpose()
        backward_layers = range(len(self.layers))
        backward_layers.reverse()
        for i in backward_layers:
            matrix = self.layers[i].update_params(matrix, learning_rate)
        return


