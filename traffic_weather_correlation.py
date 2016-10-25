from datetime import datetime, timedelta

from network_training import train_double_barreled_network, train_outer_networks
from dataPandas import read_in_streams
import os
import theano.tensor as T

time_format = '%Y-%m-%dT%H:%M'
window_duration = timedelta(hours=1)


def setup():
    datapath = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'CSV2DataStream', 'Analysis')
    main_path = os.path.join(datapath, 'traffic')
    context_path = os.path.join(datapath, 'weather')

    main_streams, main_features, context_stream, context_features = read_in_streams(main_path, context_path)

    start, end = main_streams.values()[0].get_time_range()

    context_stream.fill_in_missing_values(start, end)
    for key, stream in main_streams.items():
        # try:
        stream.fill_in_missing_values(start, end)
        #     break
        # except Exception as e:
        #     print key
        #     for key, dup in enumerate(list(stream.data.index.duplicated())):
        #         if dup:
        #             print stream.data.index[key]


    return main_streams, main_features, context_stream, context_features, start, end



def train_network(main_streams, main_features, context_stream, context_features, start, end):
    current_date = start
    # while current_date<end:
    # window_end = current_date + window_duration
    context_window = context_stream.transform_time_window_for_neural_network_input(start, end)
    for main_stream in main_streams:
        main_window = main_stream.transform_time_window_for_neural_network_input(start, end, ['avgSpeed', 'vehicleCount'])

        u, v, x_input, y_input, min_cost = train_double_barreled_network(context_window, main_window, 100, (None, 1, 2))

        loss_u, loss_v = train_outer_networks(u, v, x_input, y_input, context_window, main_window, 100)



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


if __name__ == '__main__':
    main_streams, main_features, context_stream, context_features, start, end = setup()

    train_network(main_streams.values()[:1], main_features, context_stream, context_features, start, end )





