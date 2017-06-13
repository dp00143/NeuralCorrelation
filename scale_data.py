import pandas
import scipy
import numpy
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def scale_data(dataframe, r=(-1,1)):

    scaler = MinMaxScaler(r)
    data = dataframe.values
    rescaledX = scaler.fit_transform(data)
    return rescaledX