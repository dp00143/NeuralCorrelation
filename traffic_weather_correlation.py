from datetime import datetime

from network_training import train_double_barreled_network, train_outer_networks
from dataPandas import read_in_streams
import os
from pprint import pprint

time_format = '%Y-%m-%dT%H'
current_date = datetime.strptime("2014-08-04T08", time_format)
end_date = datetime.strptime("2014-08-04T09", time_format)

datapath = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'CSV2DataStream', 'Analysis')
main_path = os.path.join(datapath, 'traffic')
context_path = os.path.join(datapath, 'weather')

main_streams, main_features, context_stream, context_features = read_in_streams(main_path, context_path)

window = context_stream.get_time_window(['tempm','wspdm'], current_date, end_date)
pprint(window)
pprint(window.values)
print window.values.shape
print 'test'
