import numpy
import pickle
import random

def sigmoid(z):
    return 1.0 / (1.0 + numpy.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def load_net(filename):
    print(f'loading NN from "{filename}"...')
    with open(filename, 'rb') as file:
        net = pickle.load(file)
    return net

class Network:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.layers = len(layer_sizes)
        self.biases = [numpy.random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [numpy.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    def save_net(self, filename):
        print(f'saving NN as "{filename}"...')
        with open(filename, 'wb') as file:
            pickle.dump(self, file)



    def feedforward(self, a):
        """ a: a numpy matrix (n ,1) representing the input """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(numpy.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None, ):
        """ training_data: a list of tuples of (input, expected result) """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_from_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}, {self.evaluate(test_data)} /  {n_test})")
            else:
                print(f"Epoch {j} complete.")

    def update_from_batch(self, batch, eta):
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = numpy.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = numpy.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    # def evaluate(self, test_data):
    #     """Return the number of test inputs for which the neural
    #     network outputs the correct result. Note that the neural
    #     network's output is assumed to be the index of whichever
    #     neuron in the final layer has the highest activation."""
    #     test_results = [(numpy.argmax(self.feedforward(x)), y) for (x, y) in test_data]
    #     return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

if __name__ == "__main__":
    # Create a network
    net = Network([2, 10, 1])

    # Create some possible input matrices
    inp00 = numpy.array([[0],[0]])
    inp10 = numpy.array([[1],[0]])
    inp01 = numpy.array([[0],[1]])
    inp11 = numpy.array([[1],[1]])

    # Zip inputs with expected outputs to create training data
    inputs = numpy.array([inp00, inp01, inp10, inp11])
    outputs = numpy.array([[1],[1],[1],[0]])
    training_data = list(zip(inputs, outputs))

    # Function for testing the output
    def test_gate(net):
        print(f"(0, 0): {net.feedforward(inp00)}")
        print(f"(0, 1): {net.feedforward(inp01)}")
        print(f"(1, 0): {net.feedforward(inp10)}")
        print(f"(1, 1): {net.feedforward(inp11)}")

    # Display the starting output
    print("Before training:")
    test_gate(net)

    # Train
    for _ in range(1000):
        net.update_from_batch(training_data, 1.0)

    # Display the final results
    print("After training:")
    test_gate(net)
