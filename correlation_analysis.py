import numpy as np
import theano
import theano.tensor as T
import time

from networks import create_neural_network
import functions

from data_generator import iterate_minibatches, generate_dataset

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


def train_double_barreled_network(x_input, y_input, num_epochs=100):

    u, u_shape, v, v_shape, x_network, y_network = double_barreled_network(x_input, y_input)

    cost = functions.minus_corr(u, v)
    x_params = lasagne.layers.get_all_params(x_network)
    y_params = lasagne.layers.get_all_params(y_network)
    x_y_params = x_params + y_params

    updates = lasagne.updates.sgd(cost, x_y_params, learning_rate=0.01)

    train_fn = theano.function([x_input, y_input], cost, updates=updates)

    print 'Start training double barreled network'


    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(x_train, y_train, 100, shuffle=True):
            x, u, y, v = batch
            train_err += train_fn(u, v)
            train_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))






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



if __name__ == '__main__':
    x_train, y_train, x_val, y_val= generate_dataset(1000)

    x_input = T.tensor3('x_inputs')
    y_input = T.tensor3('y_input')
    u_output = T.tensor3('u_output')
    v_output = T.tensor3('u_output')

    train_double_barreled_network(x_input, y_input)


