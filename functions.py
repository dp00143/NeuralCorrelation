import numpy as np
import theano
import theano.tensor as T
from scipy.stats import linregress



def minus_cor(u, v):
    u_input = T.tensor3()
    v_input = T.tensor3()
    u = np.array(u)
    v = np.array(v)

    ssxm, ssxym, ssyxm, ssym = np.cov(u, v, bias=1).flat
    r_num = ssxym
    r_den = np.sqrt(ssxm * ssym)
    if r_den == 0.0:
        r = 0.0
    else:
        r = r_num / r_den
        # test for numerical error propagation
        if (r > 1.0):
            r = 1.0
        elif (r < -1.0):
            r = -1.0

    negative_pearson = -r

    corr = theano.function([u_input, v_input], negative_pearson)

    return corr(u, v)