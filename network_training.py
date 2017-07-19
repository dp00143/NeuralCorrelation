import os
os.environ['THEANO_FLAGS'] = 'exception_verbosity=high,optimizer=fast_compile'
import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode
import time
from sklearn.model_selection import KFold

from networks import double_barreled_network, outer_network
import functions
import numpy

from data_generator import generate_sample_data, iterate_minibatches

import lasagne
import scale_data


def train_double_barreled_network(x_inputs, y_inputs, num_epochs, network_width, input_shape=(None, 1, 3), x_input=T.tensor3('x_input'), y_input=T.tensor3('y_input')):


    u, u_shape, v, v_shape, x_network, y_network = double_barreled_network(x_input, y_input, input_shape, network_width)

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


def train_outer_networks(u, v, x_input, y_input, x_inputs, y_inputs, num_epochs, corr_coefficient, network_width,
                         target_var_x=T.tensor3('x_target'), target_var_y=T.tensor3('y_target'), out_nodes=3):

    x_out, x_shape, u_network, input_var_u = outer_network(name='u_network', output_nodes=out_nodes, width=network_width)
    y_out, y_shape, v_network, input_var_v = outer_network(name='v_network', output_nodes=out_nodes, width=network_width)
    batchsize = 300

    #Training function that is used to optimize the network
    try:
        x_out_shape = (x_inputs.shape[0], x_inputs.shape[1], 1)
    except:
        x_inputs = numpy.array(x_inputs)
        y_inputs = numpy.array(y_inputs)
        x_out_shape = (x_inputs.shape[0], x_inputs.shape[1], 1)

    cost_u = lasagne.objectives.squared_error(x_out, target_var_x.reshape((x_out_shape[1], batchsize, x_out_shape[2]))).mean()
    cost_v = lasagne.objectives.squared_error(y_out, target_var_y.reshape((x_out_shape[1], batchsize, x_out_shape[2]))).mean()

    params_u = lasagne.layers.get_all_params(u_network)
    params_v = lasagne.layers.get_all_params(v_network)

    learning_rate = 0.1/16
    updates_u = lasagne.updates.nesterov_momentum(cost_u, params_u, learning_rate=learning_rate)
    updates_v = lasagne.updates.nesterov_momentum(cost_v, params_v, learning_rate=learning_rate)

    train_fn_u = theano.function([input_var_u, target_var_x], cost_u, updates=updates_u)
    train_fn_v = theano.function([input_var_v, target_var_y], cost_v, updates=updates_v)

    # Validation from training set. The crucial difference
    # here is that we do a deterministic forward pass through the network

    test_u = lasagne.layers.get_output(u_network, deterministic=True)
    test_v = lasagne.layers.get_output(v_network, deterministic=True)

    chunks = int(len(x_inputs)/batchsize)
    val_data_length = len(x_inputs)-(chunks*batchsize)
    test_cost_u = lasagne.objectives.squared_error(test_u, target_var_x.reshape((x_out_shape[1], val_data_length, x_out_shape[2]))).mean()
    test_cost_v = lasagne.objectives.squared_error(test_v, target_var_y.reshape((x_out_shape[1], val_data_length, x_out_shape[2]))).mean()
    # test_cost_u = functions.mse_target_var(test_u, target_var_x.reshape((x_out_shape[1], val_data_length, x_out_shape[2])))
    # test_cost_u = test_cost_u.mean()
    # test_cost_v = functions.mse_target_var(test_v, target_var_y.reshape((x_out_shape[1], val_data_length, x_out_shape[2])))
    # test_cost_v = test_cost_v.mean()

    val_fn_u = theano.function([input_var_u, target_var_x], test_cost_u)
    val_fn_v = theano.function([input_var_v, target_var_y], test_cost_v)


    # test_acc_u = T.mean(T.eq(T.argmax(test_u), target_var_x), dtype=theano.config.floatX)
    # test_acc_v = T.mean(T.eq(T.argmax(test_v), target_var_y), dtype=theano.config.floatX)

    # val_fn_u = theano.function([input_var_u[:, 27], target_var_x[:, 27]], [test_cost_u, test_acc_u])
    # val_fn_v = theano.function([input_var_v[:, 27], target_var_y[:, 27]], [test_cost_v, test_acc_v])



    # Finally, launch the training loop.
    print("Starting training outer Networks...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err_u = 0
        train_err_v = 0
        train_batches = 0
        start_time = time.time()

        # for start_idx in range(0, len(x_inputs) - batchsize + 1, batchsize):
        for xd, yd, _, _ in iterate_minibatches(x_inputs[:-val_data_length], y_inputs[:-val_data_length],
                                                                            x_inputs[:-val_data_length],
                                                                            y_inputs[:-val_data_length],
                                                                            batchsize, shuffle=True):
            lfaatx = [[xds[0][-2] for xds in xd]]
            lfaaty = [[yds[0][-2] for yds in yd]]
            u_eval = u.eval({x_input: xd})
            v_eval = u_eval * (-corr_coefficient)

            cur_pred = x_out.eval({input_var_u:u_eval})
            err = 0
            for cp, lf in zip(cur_pred, lfaatx[0]):
                err += (cp - lf)**2
            err = err/len(cur_pred)
            train_err_u += train_fn_u(u_eval, lfaatx)
            train_err_v += train_fn_v(v_eval, lfaaty)
            train_batches += 1

        # And a full pass over the validation data:
        # val_err_u = 0
        # val_err_v = 0
        # val_acc_u = 0
        # val_acc_v = 0
        # val_batches = 0
        # for start_idx in range(batchsize, len(x_inputs) - batchsize + 1, batchsize):

        u_eval = u.eval({x_input: x_inputs[-val_data_length:]})
        # u_eval = u.eval({x_input: x_inputs})

        # v_eval = v.eval({y_input: y_inputs[batchsize:]})
        v_eval = u_eval * (-corr_coefficient)
        # val_err_u = val_fn_u(u_eval, x_inputs)
        # val_err_v = val_fn_v(v_eval, y_inputs)
        val_x =  [[xds[0][-2]] for xds in x_inputs[-val_data_length:]]
        val_y =  [[yds[0][-2]] for yds in y_inputs[-val_data_length:]]
        val_err_u = val_fn_u(u_eval, val_x)
        val_err_v = val_fn_v(v_eval, val_y)
        # val_err_u += tmp_cost_u
        # val_err_v += tmp_cost_v
        # val_acc_u += tmp_acc_u
        # val_acc_v += tmp_acc_v

        #######
        #Predictions for T2

        x_predictions = x_out.eval({input_var_u:u_eval})
        y_predictions = y_out.eval({input_var_v:v_eval})

        squared_err = 0
        count = 0

        targets = y_inputs[-val_data_length:]
        for y_pred, target in zip(y_predictions, targets):
            squared_err += (y_pred - target[0][0]) ** 2
            # print "Current MSE:"
            # print (y_pred[-2] - target[0][-2]) ** 2
            count += 1
        squared_err = squared_err/count


            # val_batches += 1

        loss_u = train_err_u / train_batches
        loss_v = train_err_v / train_batches

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss u:\t\t{:.6f}".format(loss_u))
        print("  validation loss u:\t\t{:.6f}".format(float(val_err_u)))
        # print("  validation accuracy u:\t\t{:.2f} %".format(
        #     float(val_acc_u)))
        print("  training loss v:\t\t{:.6f}".format(float(loss_v)))
        print("  validation loss v:\t\t{:.6f}".format(float(val_err_v)))
        # print("  validation accuracy v:\t\t{:.2f} %".format(
        #       float(val_acc_v)))

        print("  validation loss on target variable:\t\t{:.6f}".format(float(squared_err)))

    return val_err_u, val_err_v, squared_err

def train_outer_networks_kfold(u, v, x_input, y_input, x_inputs, y_inputs, num_epochs, corr_coefficient, network_width,
                         target_var_x=T.tensor3('x_target'), target_var_y=T.tensor3('y_target'), kfold=10, ptids=None):
    out_variables = [-9, -7, -6]
    # out_variables = [-8, -6, -5]

    out_nodes = len(out_variables)
    x_out, x_shape, u_network, input_var_u = outer_network(name='u_network', output_nodes=out_nodes,
                                                           width=network_width)
    y_out, y_shape, v_network, input_var_v = outer_network(name='v_network', output_nodes=out_nodes,
                                                           width=network_width)
    # val_data_length = 3932
    # train_data_length = 9000
    # val_data_length = 278
    val_data_length = len(x_inputs)-1000
    train_data_length = 900
    assert val_data_length + 100 + train_data_length == len(x_inputs)


    # Training function that is used to optimize the network
    try:
        x_out_shape = (x_inputs.shape[0], x_inputs.shape[1], out_nodes)
    except:
        x_inputs = numpy.array(x_inputs)
        y_inputs = numpy.array(y_inputs)
        x_out_shape = (x_inputs.shape[0], x_inputs.shape[1], out_nodes)

    cost_u = lasagne.objectives.squared_error(x_out,
                                              target_var_x.reshape((x_out_shape[1], train_data_length, x_out_shape[2]))).mean()
    cost_v = lasagne.objectives.squared_error(y_out,
                                              target_var_y.reshape((x_out_shape[1], train_data_length, x_out_shape[2]))).mean()

    params_u = lasagne.layers.get_all_params(u_network)
    params_v = lasagne.layers.get_all_params(v_network)

    learning_rate = 0.1/16
    updates_u = lasagne.updates.nesterov_momentum(cost_u, params_u, learning_rate=learning_rate)
    updates_v = lasagne.updates.nesterov_momentum(cost_v, params_v, learning_rate=learning_rate)

    train_fn_u = theano.function([input_var_u, target_var_x], cost_u, updates=updates_u)
    train_fn_v = theano.function([input_var_v, target_var_y], cost_v, updates=updates_v)

    # Validation from training set. The crucial difference
    # here is that we do a deterministic forward pass through the network

    test_u = lasagne.layers.get_output(u_network, deterministic=True)
    test_v = lasagne.layers.get_output(v_network, deterministic=True)

    test_cost_u = lasagne.objectives.squared_error(test_u, target_var_x.reshape(
        (x_out_shape[1], val_data_length, x_out_shape[2]))).mean()
    test_cost_v = lasagne.objectives.squared_error(test_v, target_var_y.reshape(
        (x_out_shape[1], val_data_length, x_out_shape[2]))).mean()
    # test_cost_u = functions.mse_target_var(test_u, target_var_x.reshape((x_out_shape[1], val_data_length, x_out_shape[2])))
    # test_cost_u = test_cost_u.mean()
    # test_cost_v = functions.mse_target_var(test_v, target_var_y.reshape((x_out_shape[1], val_data_length, x_out_shape[2])))
    # test_cost_v = test_cost_v.mean()

    val_fn_u = theano.function([input_var_u, target_var_x], test_cost_u)
    val_fn_v = theano.function([input_var_v, target_var_y], test_cost_v)

    # test_acc_u = T.mean(T.eq(T.argmax(test_u), target_var_x), dtype=theano.config.floatX)
    # test_acc_v = T.mean(T.eq(T.argmax(test_v), target_var_y), dtype=theano.config.floatX)

    # val_fn_u = theano.function([input_var_u[:, 27], target_var_x[:, 27]], [test_cost_u, test_acc_u])
    # val_fn_v = theano.function([input_var_v[:, 27], target_var_y[:, 27]], [test_cost_v, test_acc_v])



    # Finally, launch the training loop.
    print("Starting training outer Networks...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err_u = 0
        train_err_v = 0
        train_batches = 0
        start_time = time.time()
        kf = KFold(n_splits=kfold, shuffle=True)
        folds = kf.split(x_inputs[:-val_data_length])
        for train_index, test_index in folds:
            xd_train = x_inputs[train_index]
            yd_train = y_inputs[train_index]
            xd_test = x_inputs[test_index]
            yd_test = y_inputs[test_index]
            # lfaatx = [[xds[0][-8:-1] for xds in xd_train]]
            # lfaaty = [[yds[0][-8:-1] for yds in yd_train]]
            lfaatx = [[numpy.array([xds[0][out] for out in out_variables]) for xds in xd_train]]
            lfaaty = [[numpy.array([yds[0][out] for out in out_variables]) for yds in yd_train]]
            # lfaaty = [[yds[0][-8:-1] for yds in yd_train]]

            u_eval = u.eval({x_input: xd_train})
            v_eval = u_eval * (-corr_coefficient)
            train_err_u += train_fn_u(u_eval, lfaatx)
            train_err_v += train_fn_v(v_eval, lfaaty)
            train_batches += 1

        # And a full pass over the validation data:
        # val_err_u = 0
        # val_err_v = 0
        # val_acc_u = 0
        # val_acc_v = 0
        # val_batches = 0
        # for start_idx in range(batchsize, len(x_inputs) - batchsize + 1, batchsize):

        u_eval = u.eval({x_input: x_inputs[-val_data_length:]})
        # u_eval = u.eval({x_input: x_inputs})

        # v_eval = v.eval({y_input: y_inputs[batchsize:]})
        v_eval = u_eval * (-corr_coefficient)
        # val_err_u = val_fn_u(u_eval, x_inputs)
        # val_err_v = val_fn_v(v_eval, y_inputs)
        val_x = [[numpy.array([xds[0][out] for out in out_variables])] for xds in x_inputs[-val_data_length:]]
        val_y = [[numpy.array([yds[0][out] for out in out_variables])] for yds in y_inputs[-val_data_length:]]
        val_err_u = val_fn_u(u_eval, val_x)
        val_err_v = val_fn_v(v_eval, val_y)
        # val_err_u += tmp_cost_u
        # val_err_v += tmp_cost_v
        # val_acc_u += tmp_acc_u
        # val_acc_v += tmp_acc_v

        #######
        # Predictions for T2

        x_predictions = x_out.eval({input_var_u: u_eval})
        y_predictions = y_out.eval({input_var_v: v_eval})
        if epoch == num_epochs - 1:
            actual_file = open("actual_values.csv", 'a')
            predicted_file = open("predicted_values.csv", 'a')
            actual_file.write(
                "ptid,p2,ethnicgene,partnered,p4,p10,income4grp,p25a,p25d,scqtot13t1,nummetsites,canctype,numpriortx,ex1t1,CycleLen,bpi1t2,age,p8,bmi,kpst1,yearsfromdxtostart,HGB1,gstott2,newafitott2,cesdtt2,satott2,lenat2,lenpt2,lfaat2,lfapt2\n")
            predicted_file.write('gstott2,lenat2, lenpt2,newafitott2,cesdtt2,satott2\n')
            scaled_actual_file = open("scaled_actual_values.csv", 'a')
            scaled_predicted_file = open("scaled_predicted_values.csv", 'a')
            scaled_actual_file.write("ptid,p2,ethnicgene,partnered,p4,p10,income4grp,p25a,p25d,scqtot13t1,nummetsites,canctype,numpriortx,ex1t1,CycleLen,bpi1t2,age,p8,bmi,kpst1,yearsfromdxtostart,HGB1,gstott2,newafitott2,cesdtt2,satott2,lenat2,lenpt2,lfaat2,lfapt2\n")
            scaled_predicted_file.write('gstott2,lenat2, lenpt2,newafitott2,cesdtt2,satott2\n')
        squared_err = [0 for _ in range(out_nodes)]
        targets = y_inputs[-val_data_length:]
        for y_pred, target, ptid in zip(y_predictions, targets, ptids):
            target = numpy.array([target[0][out] for out in out_variables])
            if epoch == num_epochs - 1:
                scaled_actual_values = [str(ptid)]
                scaled_actual_values.extend([str(t) for t in target])
                # scaled_actual_file.seek(0)
                scaled_actual_file.write(','.join(scaled_actual_values)+"\n")
                # scaled_predicted_file.seek(0)
                scaled_predicted_file.write(','.join([str(y) for y in y_pred]) + "\n")
            for yp, t, i, ptid in zip(y_pred, target, range(out_nodes), ptids):
                squared_err[i] += (yp-t) ** 2
        squared_err = [se / len(y_predictions) for se in squared_err]

        back_scaled_err = [0 for _ in range(out_nodes)]
        plot_yps = [[] for _ in out_variables]
        plot_targets = [[] for _ in out_variables]
        for y_pred, target, ptid in zip(y_predictions, targets, ptids):
            tmp = list(target[0])
            for out_variable, yp in zip(out_variables, y_pred):
                tmp[out_variable] = yp
            # tmp = numpy.array(tmp)
            tmp = scale_data.inverse_scale_data(tmp)
            y_pred = numpy.array([tmp[i] for i in out_variables])
            target = scale_data.inverse_scale_data(target[0])
            if epoch == num_epochs - 1:
                actual_values = [str(ptid)]
                actual_values.extend([str(t) for t in target])
                actual_file.write(','.join(actual_values) + "\n")
                predicted_file.write(','.join([str(y) for y in y_pred]) + "\n")
                target = numpy.array([target[i] for i in out_variables])
            for yp, t, i in zip(y_pred, target, range(out_nodes)):
                back_scaled_err[i] += (yp - t) ** 2
                if epoch == num_epochs - 1:
                    plot_yps[i].append(yp)
                    plot_targets[i].append(t)
        if epoch == num_epochs-1:
            from bland_altman import bland_altman_plot
            for i in range(out_nodes):
                bland_altman_plot(plot_yps[i], plot_targets[i], "feature%i.png" % i)
            actual_file.close()
            predicted_file.close()
        back_scaled_err = [se / len(y_predictions) for se in back_scaled_err]


        loss_u = train_err_u / train_batches
        loss_v = train_err_v / train_batches



        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss u:\t\t{:.6f}".format(loss_u))
        print("  validation loss u:\t\t{:.6f}".format(float(val_err_u)))
        # print("  validation accuracy u:\t\t{:.2f} %".format(
        #     float(val_acc_u)))
        print("  training loss v:\t\t{:.6f}".format(float(loss_v)))
        print("  validation loss v:\t\t{:.6f}".format(float(val_err_v)))
        # print("  validation accuracy v:\t\t{:.2f} %".format(
        #       float(val_acc_v)))

        # print("  validation loss on target variable:\t\t{:.6f}".format(float(squared_err)))
        for i, se in enumerate(squared_err):
            print("  validation loss on target variable %i: %f" % (i,float(se)))

    return val_err_u, val_err_v, squared_err, back_scaled_err

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
            if tmp[0] < loss_u:
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