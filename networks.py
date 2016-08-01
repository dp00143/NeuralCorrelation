import numpy as np
import theano
import theano.tensor as T

import lasagne
from numpy.core.numeric import outer


def create_neural_network(input_var, input_shape, output_nodes, inner_transfer_function,
                          outer_transfer_function, width=3):

    # Input layer
    in_layer = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    # Fully connected hidden layer with given transfer function
    hidden = lasagne.layers.DenseLayer(in_layer, width, nonlinearity=inner_transfer_function)

    # Fully connected output layer
    out = lasagne.layers.DenseLayer(hidden, output_nodes, nonlinearity=outer_transfer_function)

    return out

