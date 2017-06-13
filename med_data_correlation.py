import pandas
from pprint import pprint
import numpy
import theano
from network_training import train_double_barreled_network, train_outer_networks, train_outer_networks_kfold
import lasagne


def read_data():
    # to_drop = ['ptid','p2','ethnicgene','partnered','p4','p10','income4grp','p25a','p25d','scqtot13t1','nummetsites','canctype','numpriortx','ex1t1','CycleLen','age','p8','bmi','kpst1','yearsfromdxtostart','HGB1']
    to_drop = ['ptid']
    phase1 = pandas.read_csv("NikosData/Dat.T1.MaximumLikelihood.v2.csv").drop(to_drop, axis=1)
    phase2 = transform_time_window_for_neural_network_input(pandas.read_csv("NikosData/Dat.T2.MaximumLikelihood.csv").drop(to_drop, axis=1))
    columns = len(phase1.columns)
    phase1 = transform_time_window_for_neural_network_input(phase1)
    return phase1, phase2, columns

def transform_time_window_for_neural_network_input(data):
    window = numpy.array(map((lambda x: [x]), data.values))
    return window


def custom_activation4(x):
    multiplier = 5
    lin_term = 5
    return multiplier*lasagne.nonlinearities.tanh(x) - lin_term


def custom_activation8(x):
    multiplier = 60
    lin_term = 63
    return multiplier*lasagne.nonlinearities.tanh(x) - lin_term


def custom_activation6(x):
    multiplier = 27
    lin_term = 27
    return multiplier*lasagne.nonlinearities.tanh(x) - lin_term


def custom_activation5(x):
    multiplier = 30
    lin_term = 50
    return multiplier*lasagne.nonlinearities.tanh(x) - lin_term


def train(num_epochs=100, ensemble_num=20, inner_width=100, outer_width=100):
    theano.config.exception_verbosity = 'high'
    theano.config.optimizer = 'None'
    x_inputs, y_inputs, columns = read_data()

    min_cost = None
    for i in range(ensemble_num):
        if min_cost is None:
            u, v, x_input, y_input, min_cost = train_double_barreled_network(x_inputs, y_inputs, num_epochs, inner_width,
                                                                             input_shape=(None, 1, columns))
        else:
            tmp = train_double_barreled_network(x_inputs, y_inputs, num_epochs, inner_width
                                                , input_shape=(None, 1, columns))
            if tmp[4] < min_cost:
                u, v, x_input, y_input, min_cost = tmp

    # out_nodes = columns
    ses = []
    out_variables = [-8, -4, -3, -7, -6, -5]
    functions = [custom_activation8, custom_activation4, custom_activation4, custom_activation4, custom_activation6,
                 custom_activation5]
    names = ['gstott2', 'lenat2', 'lenpt2', 'newafitott2', 'cesdtt2', 'satott2']
    # out_variables = range(-8,-2)
    for out_variable, function in zip(out_variables, functions):
        out_nodes = 1
        squared_err = None

        for i in range(ensemble_num):
            if squared_err is None:
                loss_u, loss_v, squared_err = train_outer_networks_kfold(u, v, x_input, y_input, x_inputs, y_inputs, num_epochs,
                                                                         corr_coefficient=min_cost, network_width=outer_width,
                                                                         out_nodes=out_nodes, out_variable=out_variable, inner_function=function)
            else:
                tmp = train_outer_networks_kfold(u, v, x_input, y_input, x_inputs, y_inputs, num_epochs,
                                                 corr_coefficient=min_cost, network_width=outer_width, out_nodes=out_nodes, out_variable=out_variable, inner_function=function)
                if tmp[2] < squared_err:
                    loss_u, loss_v, squared_err = tmp
        ses.append((loss_u, loss_v, squared_err))


    for name, (loss_u, loss_v, squared_err) in zip(names, ses):
        # print '#####################'
        # print 'Correlation'
        # pprint(float(min_cost))
        # #
        # print '#####################'
        # print 'MSE u'
        # pprint(float(loss_u))
        #
        # print '#####################'
        # print 'MSE v'
        # pprint(float(loss_v))


        print '#####################'
        print 'MSE Target Variable %s' % name
        pprint(float(squared_err))

if __name__ == '__main__':
    train(num_epochs=100, ensemble_num=1, outer_width=150)