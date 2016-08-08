import numpy as np
import theano
import theano.tensor as T
import time

from networks import double_barreled_network, outer_network
import functions

from data_generator import iterate_minibatches, generate_dataset

import lasagne


def train_double_barreled_network(x_input, y_input, x_input_train, y_input_train, x_target_train, y_target_train, num_epochs=100):
    u, u_shape, v, v_shape, x_network, y_network = double_barreled_network(x_input, y_input)

    cost = functions.minus_corr(u, v)
    cost = cost.mean()
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
        for batch in iterate_minibatches(x_input_train, y_input_train, x_target_train, y_target_train, 100, shuffle=True):
            x, u, y, v = batch
            train_err += train_fn(u, v)
            train_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))


def train_outer_network(input_, target, name, num_epochs=100):
    out, shape, network = outer_network(inp=input_, name=name)
    cost = lasagne.objectives.squared_error(out, target)


if __name__ == '__main__':
    x_inputs_train, x_inputs_val, y_inputs_train, y_inputs_val, x_targets_train, x_targets_val, y_targets_train, \
        y_targets_val = generate_dataset(1000, 100)

    x_input = T.tensor4('x_inputs')
    y_input = T.tensor4('y_input')
    u_output = T.tensor4('u_output')
    v_output = T.tensor4('u_output')

    train_double_barreled_network(x_input, y_input, x_inputs_train, y_inputs_train, x_targets_train, y_targets_train)
