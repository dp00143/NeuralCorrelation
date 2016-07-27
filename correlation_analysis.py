import numpy as np
import theano
import theano.tensor as T
from networks import create_neural_network
import functions

import lasagne

def double_barreled_network(x_input, y_input):

    input_shape = (None, 1, 3)

    #Create double barreled neural network
    x_network = create_neural_network(x_input, input_shape=input_shape, output_nodes=1,
                                      inner_transfer_function=lasagne.nonlinearities.tanh,
                                      outer_transfer_function=lasagne.nonlinearities.linear)


    y_network = create_neural_network(y_input, input_shape=input_shape, output_nodes=1,
                                      inner_transfer_function=lasagne.nonlinearities.tanh,
                                      outer_transfer_function=lasagne.nonlinearities.linear)

    #Get output and shape of the predictions for u and v

    u_prediction = lasagne.layers.get_output(x_network)
    v_prediction = lasagne.layers.get_output(y_network)

    u_shape = lasagne.layers.get_output_shape(x_network, input_shape)
    v_shape = lasagne.layers.get_output_shape(y_network, input_shape)


    return u_prediction, u_shape, v_prediction, v_shape, x_network, y_network


def train_double_barreled_network(x_input, y_input):

    u, u_shape, v, v_shape, x_network, y_network = double_barreled_network(x_input, y_input)

    cost = functions.minus_cor(u, v)
    x_params = lasagne.layers.get_all_params(x_network)
    y_params = lasagne.layers.get_all_params(y_network)
    x_y_params = x_params + y_params

    updates = lasagne.updates.sgd(cost, x_y_params, learning_rate=0.01)

    train_fn = theano.function([])






def outer_networks(u_input, u_shape, v_input, v_shape):

    #Create both outer neural networks

    u_network = create_neural_network(u_input, input_shape=u_shape, output_nodes=3,
                                      inner_transfer_function=lasagne.nonlinearities.tanh,
                                      outer_transfer_function=lasagne.nonlinearities.linear)


    v_network = create_neural_network(v_input, input_shape=v_shape, output_nodes=3,
                                      inner_transfer_function=lasagne.nonlinearities.tanh,
                                      outer_transfer_function=lasagne.nonlinearities.linear)

    x_prediction = lasagne.layers.get_output(u_network)
    y_prediction = lasagne.layers.get_output(v_network)

    #