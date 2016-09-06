import numpy as np
import random
import theano
import theano.tensor as T
from math import sqrt


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
    # raise Exception('Not the correct way to generate data, use generate_sample_data(num=500) instead')
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
        x_targets.append([[y_input_functions[i](s) for i in range(3)]])
        y_inputs.append([[x_target_functions[i](t) for i in range(3)]])
        y_targets.append([[y_target_functions[i](s) for i in range(3)]])

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
# Functions to generate the first mode defined in Hsieh et al.
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
    return -t + 0.3 * t ** 3


def y3(t):
    return t + 0.3 * t ** 2


###################
#
# Functions to generate the second mode defined in Hsieh et al.
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
    return s + 0.3 * s ** 3


def y3_(s):
    return s - 0.3 * s ** 2



def generate_sample_data(num=500):
    t_gen = random.Random(10)
    s_gen = random.Random(20)
    x = []
    y = []
    x_modes_1 = [x1, x2, x3]
    x_modes_2 = [x1_, x2_, x3_]
    y_modes_1 = [y1, y2, y3]
    y_modes_2 = [y1_, y2_, y3_]

    for i in range(num):
        t = t_gen.uniform(-1, 1)
        s = s_gen.uniform(-(1. / sqrt(3)), 1. / sqrt(3))
        x.append([[x_modes_1[j](t) + x_modes_2[j](s) for j in range(3)]])
        # y.append([[(x_modes_1[j](t) + x_modes_2[j](s)) for j in range(3)]])
        y.append([[y_modes_1[j](s) + y_modes_2[j](s) for j in range(3)]])
    # x = add_noise(x, 490, 0.1)
    # y = add_noise(y, , 0.1)
    # x, y = normalize(x, y)
    return x, y

def add_noise(data, noise_num=50, std=0.2):
    gaus_gen = random.Random(30)
    indices = random.sample(range(1, len(data)), noise_num)
    for i, idx in zip(range(noise_num), indices):
        data[idx] = [[gaus_gen.gauss(0, std), gaus_gen.gauss(0, std), gaus_gen.gauss(0, std)]]
    return data



def generate_sample_data_for_visualisation(num=500):
    t_gen = random.Random(10)
    s_gen = random.Random(20)
    x = [[], [], []]
    y = [[], [], []]
    x_modes_1 = [x1, x2, x3]
    x_modes_2 = [x1_, x2_, x3_]
    y_modes_1 = [y1, y2, y3]
    y_modes_2 = [y1_, y2_, y3_]

    for i in range(num):
        t = t_gen.uniform(-1, 1)
        s = s_gen.uniform(-(1./sqrt(3)), 1./sqrt(3))
        for j in range(3):
            x[j].append(x_modes_1[j](t) + x_modes_2[j](s))
            y[j].append(y_modes_1[j](t) + y_modes_2[j](s))

    return x, y

def generate_theoretical_modes():
    base1 = np.linspace(-1.0, 1.0, num=100)
    base2 = np.linspace(-(1./sqrt(3)), 1./sqrt(3), num=100)
    x = [[x1(t) for t in base1], [x2(t) for t in base1], [x3(t) for t in base1]]
    y = [[y1(t) for t in base1], [y2(t) for t in base1], [y3(t) for t in base1]]

    x_ = [[x1_(t) for t in base2], [x2_(t) for t in base2], [x3_(t) for t in base2]]
    y_ = [[y1_(t) for t in base2], [y2_(t) for t in base2], [y3_(t) for t in base2]]

    # x, y, x_, y_ = normalize(x, y, x_, y_)
    return x, y, x_, y_

def normalize(*args):
    ret = []
    for ar in args:
        num_ar = np.asarray(ar)
        mean = num_ar.mean()
        std = num_ar.std()
        normalized_array = [(x-mean)/std for x in ar]
        ret.append(normalized_array)
    return ret





def plot_data_x(axis1=0, axis2=1):
    from matplotlib import pyplot as plt
    x, y = generate_sample_data_for_visualisation()
    x_mode1, y_mode1, x_mode2, y_mode2 = generate_theoretical_modes()

    plt.plot(x_mode1[axis1], x_mode1[axis2])
    plt.plot(x_mode2[axis1], x_mode2[axis2])
    plt.scatter(x[axis1], x[axis2])
    plt.xlim([-3, 2])
    plt.ylim([-3, 3])
    plt.show()
    plt.clf()

def plot_data_y(axis1=0, axis2=1):
    from matplotlib import pyplot as plt
    x, y = generate_sample_data_for_visualisation()
    x_mode1, y_mode1, x_mode2, y_mode2 = generate_theoretical_modes()

    plt.plot(y_mode1[axis1], y_mode1[axis2])
    plt.plot(y_mode2[axis1], y_mode2[axis2])
    plt.scatter(y[axis1], y[axis2])
    plt.show()
if __name__ == '__main__':
    plot_data_x(0, 1)
    plot_data_x(0, 2)
    plot_data_x(1, 2)

    plot_data_y(0, 1)
    plot_data_y(0, 2)
    plot_data_y(1, 2)

# def plot_theoretical_modes():
#     from matplotlib import pyplot as plt
#     x = np.linspace(-1.0,1.0,num=100)
#     x1 = [x1(t) for t in x]
#     x2 = [x2(t) for t in x]
#     x3 = [x3(t) for t in x]
#     y1 = [y1(t) for t in x]
#     y2 = [y2(t) for t in x]
#     y3 = [y3(t) for t in x]
#
#     x1_ = [x1_(t) for t in x]
#     x2_ = [x2_(t) for t in x]
#     x3_ = [x3_(t) for t in x]
#     y1_ = [y1_(t) for t in x]
#     y2_ = [y2_(t) for t in x]
#     y3_ = [y3_(t) for t in x]
#
#     plt.plot(x1,x2)
#     plt.plot(x1_,x2_)
#     plt.show(block=True)
#     print 'hello'
