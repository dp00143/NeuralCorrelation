import pandas
import scipy
import numpy
from sklearn.preprocessing import MinMaxScaler


def scale_data(dataframe, range=(-1,1)):

    array = dataframe.values
    # separate array into input and output components
    scaler = MinMaxScaler(feature_range=range)
    data = dataframe.values
    rescaledX = scaler.fit_transform(data)
    return rescaledX