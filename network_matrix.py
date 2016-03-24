'''
modify the mini_batch algorithm to calculate the nabla for all the vectors in
the mini_batch simultaneously

'''


import numpy as np
import random
import sys

class Network(object):
    '''
    Object representing my neural net.  takes parameter sizes which is a list
    or vector of sizes of each layer.  weights and biases are initialized by
    random
    '''
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        #Obviously don't need bias for input neurons
        np.random.seed(1)
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],
            sizes[1:]) ]

    def feedforward(self, a):
        '''
        get the output from the network given an input 'a'.
        if 'a' is in matrix form than b needs to be in diagonal matrix of
        identical weights


        '''
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        '''
        perform stochastic gradient descent:

        Training data is a list "(x,y)" of tuples where 'x' is the input and
        'y' is the desired output.  epochs is the number of iterations of SGD
        you want to run, mini_batch_size is the size of the mini batches, and
        eta is the learning rate.  if "test_data" optional variable is provided
        then the test_data will be evaulated after each epoch to guage how the
        network training is proceeding.  This is useful for determining
        progress but slows everything down...considering all the matrix dot you
        need to do to perform feedfoward(.) evaluations.
        '''

        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            #shuffle training data
            random.shuffle(training_data)
            #get the mini_batches
            mini_batches = [ training_data[k:k+mini_batch_size] for k in
                    xrange(0, n, mini_batch_size)]
            #now train our network with each mini_batch
            for mini_batch in mini_batches:
                self.mini_batch_update(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2} ".format(j,
                        self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def mini_batch_update(self, mini_batch, eta):
        """
        Update the networks weights and biases by gradient descent using
        backpropogation on a single batch
        """
        #set up the derivative matrices
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        #construct matrix of x and y
        X = np.hstack(map(lambda x: x[0], mini_batch))
        Y = np.hstack(map(lambda x: x[1], mini_batch))
        nabla_b, nabla_w = self.backprop(X,Y)
        #for x,y in mini_batch:
        #    delta_nabla_b, delta_nabla_w = self.backprop(x,y)
        #    nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b) ]
        #    nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w) ]

        #epsilon = float(1.0E-13)
        #try:
        #    assert( np.linalg.norm([np.linalg.norm(t_nabla_b[x] - nabla_b[x]) for x
        #        in range(len(nabla_b))]) < epsilon )
        #    assert( np.linalg.norm([np.linalg.norm(t_nabla_w[x] - nabla_w[x]) for x
        #        in range(len(nabla_w))]) < epsilon )
        #except AssertionError:
        #    print  np.linalg.norm([np.linalg.norm(t_nabla_b[x] - nabla_b[x]) for x
        #        in range(len(nabla_b))]) 
        #    print np.linalg.norm([np.linalg.norm(t_nabla_w[x] - nabla_w[x]) for x
        #        in range(len(nabla_w))]) 

        #now update your variables!
        self.weights = [w - (eta/len(mini_batch))*nw for w,nw in
                zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b, nb in
                zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        """
        return a tuple "(nabla_b, nabla_w)" representing the gradient of the
        cost function 'C'
        """
        #initialize lists of matrices
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #feedforward
        activation = x
        activations = [x] #list of activations
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #backwards pass
        delta = np.multiply(self.cost_derivative(activations[-1] , y) ,
        sigmoid_prime(zs[-1]) )

        nabla_b[-1] = np.sum(delta, axis=1).reshape((delta.shape[0],1))
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = np.sum(delta, axis=1).reshape((delta.shape[0],1))
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
