import pandas
from pprint import pprint
import numpy
import theano
from network_training import train_double_barreled_network, train_outer_networks, train_outer_networks_kfold
from scale_data import scale_data

def read_data(i=None):
    to_drop = ['ptid','p2','ethnicgene','partnered','p4','p10','income4grp','p25a','p25d','scqtot13t1','nummetsites','canctypeDum','numpriortx','ex1t1','CycleLen','age','p8','bmi','kpst1','yearsfromdxtostart','HGB1']
    # p1_name = "NikosData/Dat.T1.MultipleImputation.v2.csv"
    # p2_name = "NikosData/Dat.T2.MultipleImputation.v2.csv"
    p1_name = "NikosData/Dat.T1.with.MissingFiles.v2.csv"
    p2_name = "NikosData/Dat.T2.with.MissingFiles.v2.csv"

    # p1_name = "NikosData/Dat.T1.MaximumLikelihood.v2.csv"
    # p2_name = "NikosData/Dat.T2.MaximumLikelihood.v2.csv"
    p1 = pandas.read_csv(p1_name)
    ptids = p1['ptid'].values[-278:]
    # to_drop = ['ptid']
    phase1 = pandas.read_csv(p1_name).drop(to_drop, axis=1)
    # phase1 = phase1.sample(frac=1).reset_index(drop=True)
    columns = len(phase1.columns)
    phase2 = pandas.read_csv(p2_name).drop(to_drop, axis=1)
    phase1 = phase1.dropna(axis=0, how='any')
    phase2 = phase2.dropna(axis=0, how='any')

    if i is not None:
        phase1 = phase1[phase1['Imputation_'] == i]
        phase2 = phase2[phase2['Imputation_'] == i]

    phase1 = transform_time_window_for_neural_network_input(phase1)
    phase2 = transform_time_window_for_neural_network_input(phase2)



    return phase1, phase2, columns, ptids

def transform_time_window_for_neural_network_input(data):
    data = scale_data(data)
    window = numpy.array(map((lambda x: [x]), data))
    return window


def train(num_epochs=100, ensemble_num=20, inner_width=100, outer_width=100):
    theano.config.exception_verbosity = 'high'
    theano.config.optimizer = 'None'
    x_inputs, y_inputs, columns, ptids = read_data()

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
    # num_epochs = 10000
    # ensemble_num = 20

    back_scaled_error = None
    for i in range(ensemble_num):
        if back_scaled_error is None:
            loss_u, loss_v, squared_err, back_scaled_error = train_outer_networks_kfold(u, v, x_input, y_input, x_inputs, y_inputs, num_epochs,
                                                               corr_coefficient=min_cost,  network_width=outer_width,
                                                               ptids=ptids)
        else:
            tmp = train_outer_networks_kfold(u, v, x_input, y_input, x_inputs, y_inputs, num_epochs,
                                       corr_coefficient=min_cost,  network_width=outer_width, ptids=ptids)
            better = True
            for t, b in zip(tmp[3], back_scaled_error):
                better = True and t < b
            if better:
                loss_u, loss_v, squared_err, back_scaled_error = tmp



    print '#####################'
    print 'Correlation'
    pprint(float(min_cost))
    #
    print '#####################'
    print 'MSE u'
    pprint(float(loss_u))

    print '#####################'
    print 'MSE v'
    pprint(float(loss_v))


    # print '#####################'
    # print 'MSE Target Variables total'
    # pprint(float(squared_err))


    print '#####################'
    # print 'MSE Target Variables mean'
    # pprint(float(squared_err)/out_nodes)
    for i, se in enumerate(squared_err):
        print("  MSE on target variable %i: %f" % (i, float(se)))
    for i, se in enumerate(back_scaled_error):
        print("  MSE back scaled on target variable %i: %f" % (i, float(se)))
if __name__ == '__main__':
    train(num_epochs=100, ensemble_num=20, inner_width=50, outer_width=50)