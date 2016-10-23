from datetime import datetime

from network_training import train_double_barreled_network, train_outer_networks
from dataPandas import read_in_streams
import os
from pprint import pprint

time_format = '%Y-%m-%dT%H:%M'
# current_date = datetime.strptime("2014-08-01T00:00", time_format)
# end_date = datetime.strptime("2014-08-04T09:00", time_format)

datapath = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'CSV2DataStream', 'Analysis')
main_path = os.path.join(datapath, 'traffic')
context_path = os.path.join(datapath, 'weather')

main_streams, main_features, context_stream, context_features = read_in_streams(main_path, context_path)

start, end = main_streams.values()[0].get_time_range()
window_end = datetime.strptime("2014-08-04T09:00", time_format)

context_stream.fill_in_missing_values(start, end)
for key, stream in main_streams.items():
    stream.fill_in_missing_values(start, end)
window = context_stream.transform_time_window_for_neural_network_input(start, window_end)
# pprint(window)
# pprint(window.values)
print window.values.shape
# point = context_stream.get_point_in_time(current_date)
# pprint(point)
print 'test'
