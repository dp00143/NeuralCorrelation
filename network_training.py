import os
os.environ['THEANO_FLAGS'] = 'exception_verbosity=high,optimizer=fast_compile'
import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode
import time

from networks import double_barreled_network, outer_network
import functions
import numpy

from data_generator import generate_sample_data

import lasagne


def train_double_barreled_network(x_inputs, y_inputs, num_epochs, input_shape=(None, 1, 3)):
    x_input = T.tensor3('x_input')
    y_input = T.tensor3('y_input')

    u, u_shape, v, v_shape, x_network, y_network = double_barreled_network(x_input, y_input, input_shape)

    cost = functions.minus_corr(u, v)
    cost = cost.mean()
    x_params = lasagne.layers.get_all_params(x_network)
    y_params = lasagne.layers.get_all_params(y_network)
    x_y_params = x_params + y_params

    updates = lasagne.updates.nesterov_momentum(cost, x_y_params, learning_rate=0.1/16)


    # train_fn = theano.function([x_input, y_input, u, v], cost, updates=updates)

    train_fn = theano.function([x_input, y_input], cost, updates=updates)
                               # ,
                               # mode=NanGuardMode(nan_is_error=True, inf_is_error=True,
                               #                                               big_is_error=True))

    print 'Start training double barreled network'

    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        batchsize = 500
        for start_idx in range(0, len(x_inputs) - batchsize + 1, batchsize):
            curr_err = train_fn(x_inputs[start_idx:start_idx+batchsize], y_inputs[start_idx:start_idx+batchsize])
            train_err += curr_err
            train_batches += 1

        training_loss = train_err / train_batches
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.20f}".format(training_loss))

    return u, v, x_input, y_input, training_loss


def train_outer_networks(u, v, x_input, y_input, x_inputs, y_inputs, num_epochs,
                         target_var_x=T.tensor3('x_target'), target_var_y=T.tensor3('y_target')):

    x_out, x_shape, u_network, input_var_u = outer_network(name='u_network', output_nodes=3)
    y_out, y_shape, v_network, input_var_v = outer_network(name='v_network', output_nodes=3)
    batchsize = 500

    #Training function that is used to optimize the network
    try:
        x_out_shape = x_inputs.shape
    except:
        x_inputs = numpy.array(x_inputs)
        y_inputs = numpy.array(y_inputs)
        x_out_shape = x_inputs.shape

    cost_u = lasagne.objectives.squared_error(x_out, target_var_x.reshape((x_out_shape[1], batchsize, x_out_shape[2]))).mean()
    cost_v = lasagne.objectives.squared_error(y_out, target_var_y.reshape((x_out_shape[1], batchsize, x_out_shape[2]))).mean()

    params_u = lasagne.layers.get_all_params(u_network)
    params_v = lasagne.layers.get_all_params(v_network)

    updates_u = lasagne.updates.nesterov_momentum(cost_u, params_u, learning_rate=0.1)
    updates_v = lasagne.updates.nesterov_momentum(cost_v, params_v, learning_rate=0.1)

    train_fn_u = theano.function([input_var_u, target_var_x], cost_u, updates=updates_u)
    train_fn_v = theano.function([input_var_v, target_var_y], cost_v, updates=updates_v)

    # # Validation from training set. The crucial difference
    # # here is that we do a deterministic forward pass through the network
    #
    # test_u = lasagne.layers.get_output(u_network, deterministic=True)
    # test_v = lasagne.layers.get_output(v_network, deterministic=True)
    #
    # test_cost_u = lasagne.objectives.squared_error(x_out, target_var_x.reshape((1, 500, 3))).mean()
    # test_cost_v = lasagne.objectives.squared_error(y_out, target_var_y.reshape((1, 500, 3))).mean()
    #
    # test_acc_u = T.mean(T.eq(T.argmax(test_u), target_var_x), dtype=theano.config.floatX)
    # test_acc_v = T.mean(T.eq(T.argmax(test_v), target_var_y), dtype=theano.config.floatX)
    #
    # val_fn_u = theano.function([input_var_u, target_var_x], [test_cost_u, test_acc_u])
    # val_fn_v = theano.function([input_var_v, target_var_y], [test_cost_v, test_acc_v])



    # Finally, launch the training loop.
    print("Starting training outer Networks...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err_u = 0
        train_err_v = 0
        train_batches = 0
        start_time = time.time()

        for start_idx in range(0, len(x_inputs) - batchsize + 1, batchsize):
            u_eval = u.eval({x_input: x_inputs[start_idx:start_idx + batchsize]})
            v_eval = v.eval({y_input: y_inputs[start_idx:start_idx + batchsize]})
            train_err_u += train_fn_u(u_eval, x_inputs[start_idx:start_idx + batchsize])
            train_err_v += train_fn_v(v_eval, y_inputs[start_idx:start_idx + batchsize])
            train_batches += 1

        # And a full pass over the validation data:
        # val_err_u = 0
        # val_err_v = 0
        # val_acc_u = 0
        # val_acc_v = 0
        # val_batches = 0
        # for batch in iterate_minibatches(x_inputs_val, x_targets_val, y_inputs_val, y_targets_val, 500, shuffle=False):
        #     x, x1, y, y1 = batch
        #     u_eval = u.eval({x_input: x})
        #     v_eval = v.eval({y_input: y})
        #     tmp_cost_u, tmp_acc_u = val_fn_u(u_eval, x1)
        #     tmp_cost_v, tmp_acc_v = val_fn_v(v_eval, x1)
        #     val_err_u += tmp_cost_u
        #     val_err_v += tmp_cost_v
        #     val_acc_u += tmp_acc_u
        #     val_acc_v += tmp_acc_v
        #
        #     val_batches += 1

        loss_u = train_err_u / train_batches
        loss_v = train_err_v / train_batches

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss u:\t\t{:.6f}".format(loss_u))
        # print("  validation loss u:\t\t{:.6f}".format(val_err_u / val_batches))
        # print("  validation accuracy u:\t\t{:.2f} %".format(
        #     val_acc_u / val_batches * 100))
        print("  training loss v:\t\t{:.6f}".format(loss_v))
        # print("  validation loss v:\t\t{:.6f}".format(val_err_v / val_batches))
        # print("  validation accuracy v:\t\t{:.2f} %".format(
        #       val_acc_v / val_batches * 100))

    return loss_u, loss_v

if __name__ == '__main__':
    theano.config.exception_verbosity='high'
    theano.config.optimizer='None'
    x_inputs, y_inputs = generate_sample_data()

    num_epochs = 100

    min_cost = None

    ensemble_num = 1

    for i in range(ensemble_num):
        if min_cost is None:
            u, v, x_input, y_input, min_cost = train_double_barreled_network(x_inputs, y_inputs, num_epochs)
        else:
            tmp = train_double_barreled_network(x_inputs, y_inputs, num_epochs)
            if tmp[4] < min_cost:
                u, v, x_input, y_input, min_cost = tmp



    loss_u, loss_v = None, None
    for i in range(ensemble_num):
        if loss_u is None:
            loss_u, loss_v = train_outer_networks(u, v, x_input, y_input, x_inputs, y_inputs, num_epochs)
        else:
            tmp = train_outer_networks(u, v, x_input, y_input, x_inputs, y_inputs, num_epochs)
            if tmp[0]  < loss_u:
                loss_u, loss_v = tmp

    # loss_v = None
    # for i in range(ensemble_num):
    #     if loss_v is None:
    #         _, loss_v = train_outer_networks(u, v, x_input, y_input, x_inputs, y_inputs, num_epochs)
    #     else:
    #         tmp = train_outer_networks(u, v, x_input, y_input, x_inputs, y_inputs, num_epochs)
    #         if tmp[1] < loss_v:
    #             _, loss_v = tmp

    from pprint import pprint
    print '#####################'
    print 'Correlation'
    pprint(min_cost)
    #
    print '#####################'
    print 'MSE u'
    pprint(loss_u)

    print '#####################'
    print 'MSE v'
    pprint(loss_v)
    
    #
    # pprint(x_inputs)
    # pprint(y_inputs)
    #
    # u_eval = u.eval({x_input: x_inputs})
    # v_eval = v.eval({y_input: y_inputs})



    # pprint(u_eval)
    # pprint(v_eval)