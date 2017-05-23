import numpy as np
import theano
import theano.tensor as T

import lasagne
from numpy.core.numeric import outer


def create_neural_network(input_var, input_shape, output_nodes, inner_transfer_function, name,
                          outer_transfer_function, width):

    # Input layer
    in_layer = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var, name=name)

    # Fully connected hidden layer with given transfer function
    hidden = lasagne.layers.DenseLayer(in_layer, width, nonlinearity=inner_transfer_function, name='%s.hidden' % name)

    # Fully connected output layer
    out = lasagne.layers.DenseLayer(hidden, output_nodes, nonlinearity=outer_transfer_function, name='%s.out' % name)

    if input_var is None:
        input_var = in_layer.input_var
        return out, input_var
    else:
        return out

def double_barreled_network(x_input, y_input, input_shape, width):

    #Create double barreled neural network
    x_network = create_neural_network(x_input, name='x_network', input_shape=input_shape, output_nodes=1,
                                      inner_transfer_function=lasagne.nonlinearities.tanh,
                                      outer_transfer_function=lasagne.nonlinearities.linear, width=width)


    y_network = create_neural_network(y_input, name='y_network', input_shape=input_shape, output_nodes=1,
                                      inner_transfer_function=lasagne.nonlinearities.tanh,
                                      outer_transfer_function=lasagne.nonlinearities.linear, width=width)

    #Get output and shape of the predictions for u and v

    u_prediction = lasagne.layers.get_output(x_network)
    v_prediction = lasagne.layers.get_output(y_network)

    # u_prediction = x_network.get_output_for(x_input)
    # v_prediction = y_network.get_output_for(y_input)

    u_shape = lasagne.layers.get_output_shape(x_network, input_shape)
    v_shape = lasagne.layers.get_output_shape(y_network, input_shape)
    # u_shape = x_network.get_output_shape_for(x_input)
    # v_shape = y_network.get_output_shape_for(y_input)

    return u_prediction, u_shape, v_prediction, v_shape, x_network, y_network


def outer_network(name, output_nodes, width):

    input_shape = (None, 1)
    # input_shape = (None, inp.shape)
    network, input_var = create_neural_network(None, name=name, input_shape=input_shape, output_nodes=output_nodes,
                                    inner_transfer_function=lasagne.nonlinearities.tanh,
                                    outer_transfer_function=lasagne.nonlinearities.linear, width=width)

    prediction = lasagne.layers.get_output(network)
    shape = lasagne.layers.get_output_shape(network)

    return prediction, shape, network, input_var