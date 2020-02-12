import numpy
#import random

def sigmoid(z):
    return 1.0 / (1.0 + numpy.exp(-z))

class Network:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.layers = len(layer_sizes)
        self.biases = [numpy.random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [numpy.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    def feedforward(self, a):
        """ a: a numpy matrix (n ,1) representing the input """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(numpy.dot(w, a) + b)
        return a


net = Network([2,5,1])
inp00 = numpy.array([[0],[0]])
inp10 = numpy.array([[1],[0]])
inp01 = numpy.array([[0],[1]])
inp11 = numpy.array([[1],[1]])
print(net.feedforward(inp00))
print(net.feedforward(inp01))
print(net.feedforward(inp10))
print(net.feedforward(inp11))
# for i in range(net.layers-1):
#     print(net.biases[i], net.weights[i])
#     print("---")
# print(net.weights[0])

# print(sigmoid(net.weights[0]))
