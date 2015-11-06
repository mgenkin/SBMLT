#Sigmoid Neurons
import math
'''Sigmoid Neuron Class'''
class Sigmoid:
	def __init__(self, threshold, weight):
		self.b = threshold
		self.w = weight
		self.DEBUG = True

	def __len__(self):
		return len(self.w)

	'''Calculate if the sum of w_i * INPUT_i is > b'''
	def output(self, input):
		#if input length does not match raise an error
		if(len(input) != len(self.w)):
			raise IOError

		#Debug
		if self.DEBUG:
			print "===================================output(self, input)"
			print "Current weight:", self.w
			print "Length of input:", len(input)
			print "Input array:", input

		#Calculate the SUM_i (W_i * INPUT_i)
		i = range(len(input))
		T = 0##total
		for i in input:
			T += self.w[i] * input[i]

		#Debug
		if self.DEBUG:
			print "Output of the function:", T

		#result of sigmoid function
		V = 1 / (1 + math.exp(-T - self.b))

		#Debug
		if self.DEBUG:
			print "Result of sigmoid function is", V
			
		if V > self.b:
			return True
		else:
			return False

#Frame for unit testing
def unitTest():
	input = [1, 0, 0, 0]
	sNode = Sigmoid(0.9, [0.25, 0.25, 0.25, 0.25])
	print sNode.output(input)

unitTest()