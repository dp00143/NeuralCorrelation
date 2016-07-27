import numpy as np
import theano
import theano.tensor as T
from scipy.stats import linregress



def minus_cor(u, v):
    negative_pearson = -linregress(u, v)[2]

    return negative_pearson