import numpy as np
import random
import theano
import theano.tensor as T


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.
#
# This Function was taken from the lasagne tutorial file mnist.py and adapted to accomodate two in- and output variables
#


def iterate_minibatches(x_inputs, x_targets, y_inputs, y_targets, batchsize, shuffle=False):
    assert len(x_inputs) == len(x_targets) and len(y_inputs) == len(y_targets)
    if shuffle:
        indices = np.arange(len(x_inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(x_inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield x_inputs[excerpt], x_targets[excerpt], y_inputs[excerpt], y_targets[excerpt]


def generate_dataset(n, validation_n):
    rng = random.Random(123)

    x_inputs, y_inputs, x_targets, y_targets = [], [], [], []
    x_input_functions = [x1, x2, x3]
    y_input_functions = [y1, y2, y3]
    x_target_functions = [x1_, x2_, x3_]
    y_target_functions = [y1_, y1_, y1_]

    for i in range(n):
        s = rng.uniform(-1, 1)
        t = rng.uniform(-1, 1)

        x_inputs.append([[x_input_functions[i](t) for i in range(3)]])
        x_targets.append([[y_input_functions[i](t) for i in range(3)]])
        y_inputs.append([[x_target_functions[i](t) for i in range(3)]])
        y_targets.append([[y_target_functions[i](t) for i in range(3)]])

    x_inputs_train, x_inputs_val = np.array(x_inputs[:-validation_n]), np.array(x_inputs[-validation_n:])
    y_inputs_train, y_inputs_val = np.array(y_inputs[:-validation_n]), np.array(y_inputs[-validation_n:])
    x_targets_train, x_targets_val = np.array(x_targets[:-validation_n]), np.array(x_targets[-validation_n:])
    y_targets_train, y_targets_val = np.array(y_targets[:-validation_n]), np.array(y_targets[-validation_n:])
    # x_inputs_train, x_inputs_val = [x_inputs[i][:-validation_n] for i in range(3)], \
    #                                [x_inputs[i][-validation_n:] for i in range(3)]
    # y_inputs_train, y_inputs_val = [y_inputs[i][:-validation_n] for i in range(3)], \
    #                                [x_inputs[i][-validation_n:] for i in range(3)]
    # x_targets_train, x_targets_val = [x_targets[i][:-validation_n] for i in range(3)], \
    #                                  [x_inputs[i][-validation_n:] for i in range(3)]
    # y_targets_train, y_targets_val = [y_targets[i][:-validation_n] for i in range(3)], \
    #                                  [x_inputs[i][-validation_n:] for i in range(3)]

    return x_inputs_train, x_inputs_val, y_inputs_train, y_inputs_val, x_targets_train, x_targets_val, y_targets_train, y_targets_val


###################
#
# Functions to generate the sample inputs defined in Hsieh et al.
#


def x1(t):
    return t - 0.3 * t ** 2


def x2(t):
    return t + 0.3 * t ** 3


def x3(t):
    return t ** 2


def y1(t):
    return t ** 3


def y2(t):
    return -t + 0.3 ** 3


def y3(t):
    return t + 0.3 * t ** 2


###################
#
# Functions to generate the sample targets defined in Hsieh et al.
#

def x1_(s):
    return -s - 0.3 * s ** 2


def x2_(s):
    return s - 0.3 * s ** 3


def x3_(s):
    return -s ** 4


def y1_(s):
    return 1. / np.cosh(4 * s)


def y2_(s):
    return s + 0.3 ** 3


def y3_(s):
    return s - 0.3 * s ** 2
