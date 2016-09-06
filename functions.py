import numpy as np
import theano
import theano.tensor as T
from scipy.stats import linregress
from pprint import pprint


def cosine_similarity(u, v):
    inner = T.dot(u.T, v)
    u_norm = T.sqrt(T.sum(T.sqr(u)))
    v_norm = T.sqrt(T.sum(T.sqr(v)))
    return inner / u_norm * v_norm


def minus_corr(u, v):

    #First give expression calculating the cosine similarity
    # cos_sim = cosine_similarity(u - T.mean(u), v - T.mean(u))

    #Cosine similarity can be converted to Pearson coefficient (Corr(x,y) = CosSim(x-x_mean, y-y_mean)
    corr = cosine_similarity(u-T.mean(u), v-T.mean(v))
    neg_corr = -corr
    # neg_corr = T.clip(neg_corr, -1, 1)

    return neg_corr

def minus_corr_with_constraints(u, v):
    neg_corr= minus_corr(u, v)
    avg_u = T.mean(u)
    avg_v = T.mean(v)
    avg_squared_u = T.mean(T.dot(u.T, u))
    avg_squared_v = T.mean(T.dot(v.T, v))

    res = neg_corr + (avg_u)**2 + (avg_v)**2 + ((T.sqrt(avg_squared_u)-1))**2+((T.sqrt(avg_squared_v)-1))**2
    return res