import numpy as np
import theano
import theano.tensor as T
import time

from networks import double_barreled_network, outer_network
import functions

from data_generator import iterate_minibatches, generate_dataset

import lasagne


def train_double_barreled_network(x_input_train, y_input_train, x_target_train, y_target_train, num_epochs):
    x_input = T.tensor3('x_input')
    y_input = T.tensor3('y_input')

    u, u_shape, v, v_shape, x_network, y_network = double_barreled_network(x_input, y_input)

    cost = functions.minus_corr(u, v).mean()
    x_params = lasagne.layers.get_all_params(x_network)
    y_params = lasagne.layers.get_all_params(y_network)
    x_y_params = x_params + y_params

    updates = lasagne.updates.sgd(cost, x_y_params, learning_rate=0.01)

    train_fn = theano.function([x_input, y_input, u, v], cost, updates=updates)

    print 'Start training double barreled network'

    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(x_input_train, y_input_train, x_target_train, y_target_train, 100, shuffle=True):
            x, x1, y, y1 = batch
            u_eval = u.eval({x_input: x})
            v_eval = v.eval({y_input: y})
            train_err += train_fn(x, y, u_eval, v_eval)
            train_batches += 1


        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

    return u, v, x_input, y_input


def train_outer_networks(u, v, x_input, y_input, x_input_train, y_input_train, x_target_train, y_target_train, num_epochs):
    # input_var_u = T.ivector('u_input')
    # input_var_v = T.tensor('v_input')
    # tensor.Tensor(dtype=_coding_dist.dtype,
    #               broadcastable=[False])

    target_var_x = T.tensor3('x_target')
    target_var_y = T.tensor3('y_target')
    # theano.tensor.nnet.categorical_crossentropy()
    x_out, x_shape, u_network, input_var_u = outer_network(name='u_network')
    y_out, y_shape, v_network, input_var_v = outer_network(name='v_network')




    cost_u = lasagne.objectives.squared_error(x_out, target_var_x.reshape((1,100,3))).mean()
    cost_v = lasagne.objectives.squared_error(y_out, target_var_y.reshape((1,100,3))).mean()

    params_u = lasagne.layers.get_all_params(u_network)
    params_v = lasagne.layers.get_all_params(v_network)

    updates_u = lasagne.updates.sgd(cost_u, params_u, learning_rate=0.1)
    updates_v = lasagne.updates.sgd(cost_v, params_v, learning_rate=0.1)

    # train_fn_u = theano.function([input_var_u, x_out, target_var_x], cost_u, updates=updates_u)
    # train_fn_v = theano.function([input_var_v, y_out, target_var_y], cost_v, updates=updates_v)

    train_fn_u = theano.function([input_var_u, target_var_x], cost_u, updates=updates_u)
    train_fn_v = theano.function([input_var_v, target_var_y], cost_v, updates=updates_v)


    # Finally, launch the training loop.
    print("Starting training outer Networks...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err_u = 0
        train_err_v = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(x_input_train, y_input_train, x_target_train, y_target_train, 100, shuffle=True):
            x, x1, y, y1 = batch
            u_eval = u.eval({x_input: x})
            v_eval = v.eval({y_input: y})
            x_eval = x_out.eval({input_var_u: u_eval})
            y_eval = y_out.eval({input_var_v: v_eval})
            # train_err_u += train_fn_u(u_eval, x_eval, x1)
            # train_err_v += train_fn_v(v_eval, y_eval, y1)
            train_err_u += train_fn_u(u_eval, x1)
            train_err_v += train_fn_v(v_eval, y1)
            train_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss u:\t\t{:.6f}".format(train_err_u / train_batches))
        print("  training loss v:\t\t{:.6f}".format(train_err_v / train_batches))


if __name__ == '__main__':
    impossible = True
    possible = impossible
    theano.config.exception_verbosity='high'
    theano.config.optimizer='None'
    x_inputs_train, x_inputs_val, y_inputs_train, y_inputs_val, x_targets_train, x_targets_val, y_targets_train, \
        y_targets_val = generate_dataset(100000, 10000)

    num_epochs = 1000

    u, v, x_input, y_input = train_double_barreled_network(x_inputs_train, y_inputs_train, x_targets_train, y_targets_train, num_epochs)

    train_outer_networks(u, v, x_input, y_input, x_inputs_train, y_inputs_train, x_targets_train, y_targets_train, num_epochs)