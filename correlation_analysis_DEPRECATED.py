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
        print("  training loss:\t\t{:.20f}".format(train_err / train_batches))

    return u, v, x_input, y_input


def train_outer_networks(u, v, x_input, y_input, x_input_train, y_input_train, x_target_train, y_target_train,
                         x_inputs_val, x_targets_val, y_inputs_val, y_targets_val, num_epochs):
    # input_var_u = T.ivector('u_input')
    # input_var_v = T.tensor('v_input')
    # tensor.Tensor(dtype=_coding_dist.dtype,
    #               broadcastable=[False])

    target_var_x = T.tensor3('x_target')
    target_var_y = T.tensor3('y_target')
    # theano.tensor.nnet.categorical_crossentropy()
    x_out, x_shape, u_network, input_var_u = outer_network(name='u_network')
    y_out, y_shape, v_network, input_var_v = outer_network(name='v_network')


    #Training function that is used to optimize the network

    cost_u = lasagne.objectives.squared_error(x_out, target_var_x.reshape((1,500,3))).mean()
    cost_v = lasagne.objectives.squared_error(y_out, target_var_y.reshape((1,500,3))).mean()

    params_u = lasagne.layers.get_all_params(u_network)
    params_v = lasagne.layers.get_all_params(v_network)

    updates_u = lasagne.updates.sgd(cost_u, params_u, learning_rate=0.1)
    updates_v = lasagne.updates.sgd(cost_v, params_v, learning_rate=0.1)

    train_fn_u = theano.function([input_var_u, target_var_x], cost_u, updates=updates_u)
    train_fn_v = theano.function([input_var_v, target_var_y], cost_v, updates=updates_v)

    # Validation from training set. The crucial difference
    # here is that we do a deterministic forward pass through the network

    test_u = lasagne.layers.get_output(u_network, deterministic=True)
    test_v = lasagne.layers.get_output(v_network, deterministic=True)

    test_cost_u = lasagne.objectives.squared_error(x_out, target_var_x.reshape((1, 500, 3))).mean()
    test_cost_v = lasagne.objectives.squared_error(y_out, target_var_y.reshape((1, 500, 3))).mean()

    test_acc_u = T.mean(T.eq(T.argmax(test_u), target_var_x), dtype=theano.config.floatX)
    test_acc_v = T.mean(T.eq(T.argmax(test_v), target_var_y), dtype=theano.config.floatX)

    val_fn_u = theano.function([input_var_u, target_var_x], [test_cost_u, test_acc_u])
    val_fn_v = theano.function([input_var_v, target_var_y], [test_cost_v, test_acc_v])

    # Accuracy function




    # Finally, launch the training loop.
    print("Starting training outer Networks...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err_u = 0
        train_err_v = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(x_input_train, y_input_train, x_target_train, y_target_train, 500, shuffle=True):
            x, x1, y, y1 = batch
            u_eval = u.eval({x_input: x})
            v_eval = v.eval({y_input: y})
            train_err_u += train_fn_u(u_eval, x1)
            train_err_v += train_fn_v(v_eval, y1)
            train_batches += 1

        # And a full pass over the validation data:
        val_err_u = 0
        val_err_v = 0
        val_acc_u = 0
        val_acc_v = 0
        val_batches = 0
        for batch in iterate_minibatches(x_inputs_val, x_targets_val, y_inputs_val, y_targets_val, 500, shuffle=False):
            x, x1, y, y1 = batch
            u_eval = u.eval({x_input: x})
            v_eval = v.eval({y_input: y})
            tmp_cost_u, tmp_acc_u = val_fn_u(u_eval, x1)
            tmp_cost_v, tmp_acc_v = val_fn_v(v_eval, x1)
            val_err_u += tmp_cost_u
            val_err_v += tmp_cost_v
            val_acc_u += tmp_acc_u
            val_acc_v += tmp_acc_v

            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss u:\t\t{:.6f}".format(train_err_u / train_batches))
        print("  validation loss u:\t\t{:.6f}".format(val_err_u / val_batches))
        print("  validation accuracy u:\t\t{:.2f} %".format(
            val_acc_u / val_batches * 100))
        print("  training loss v:\t\t{:.6f}".format(train_err_v / train_batches))
        print("  validation loss v:\t\t{:.6f}".format(val_err_v / val_batches))
        print("  validation accuracy v:\t\t{:.2f} %".format(
              val_acc_v / val_batches * 100))

if __name__ == '__main__':
    theano.config.exception_verbosity='high'
    theano.config.optimizer='None'
    x_inputs_train, x_inputs_val, y_inputs_train, y_inputs_val, x_targets_train, x_targets_val, y_targets_train, \
        y_targets_val = generate_dataset(1000000, 100000)

    num_epochs = 1000

    u, v, x_input, y_input = train_double_barreled_network(x_inputs_train, y_inputs_train, x_targets_train, y_targets_train,
                                                           num_epochs)

    train_outer_networks(u, v, x_input, y_input, x_inputs_train, y_inputs_train, x_targets_train, y_targets_train,
                         x_inputs_val, x_targets_val, y_inputs_val, y_targets_val, num_epochs)