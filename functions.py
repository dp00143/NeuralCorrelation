import numpy as np
import theano
import theano.tensor as T
from scipy.stats import linregress
from pprint import pprint




def minus_corr(u, v):
    um = T.sub(u, T.mean(u))
    vm = T.sub(v, T.mean(v))
    r_num = T.sum(T.mul(um, vm))
    r_den = T.sqrt(T.mul(T.sum(T.sqr(um)), T.sum(T.sqr(vm))))
    r = T.true_div(r_num, r_den)
    r = T.neg(r)
    return r

def minus_corr_with_constraints(u, v):
    neg_corr= minus_corr(u, v)
    avg_u = T.mean(u)
    avg_v = T.mean(v)
    avg_squared_u = T.mean(T.dot(u.T, u))
    avg_squared_v = T.mean(T.dot(v.T, v))

    res = neg_corr + (avg_u)**2 + (avg_v)**2 + ((T.sqrt(avg_squared_u)-1))**2+((T.sqrt(avg_squared_v)-1))**2
    return res

def mse_target_var(a, b):
    return (a[-2] - b[0][-2])**2