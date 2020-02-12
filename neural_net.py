import numpy
import random

class Network:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.layers = len(layer_sizes)
        self.biases = [numpy.random.randn(y, 1) for y in layer_sizes[1:]]


net = Network([3,5,2])
print(net.biases)

