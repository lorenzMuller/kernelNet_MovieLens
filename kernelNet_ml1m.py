'''
written by Lorenz Muller
'''

import numpy as np
import tensorflow as tf
from time import time
import sys
from dataLoader import loadData
import os

seed = int(time())
np.random.seed(seed)


# load data
tr, vr = loadData('./ml-1m/ratings.dat', delimiter='::',
                  seed=seed, transpose=True, valfrac=0.1)

tm = np.greater(tr, 1e-12).astype('float32')  # masks indicating non-zero entries
vm = np.greater(vr, 1e-12).astype('float32')

n_m = tr.shape[0]  # number of movies
n_u = tr.shape[1]  # number of users (may be switched depending on 'transpose' in loadData)

# Set hyper-parameters
n_hid = 500
lambda_2 = float(sys.argv[1]) if len(sys.argv) > 1 else 60.
lambda_s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.013
n_layers = 2
output_every = 50  # evaluate performance on test set; breaks l-bfgs loop
n_epoch = n_layers * 10 * output_every
verbose_bfgs = True
use_gpu = True
if not use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
# Input placeholders
R = tf.placeholder("float", [None, n_u])


# define network functions
def kernel(u, v):
    """
    Sparsifying kernel function

    :param u: input vectors [n_in, 1, n_dim]
    :param v: output vectors [1, n_hid, n_dim]
    :return: input to output connection matrix
    """
    dist = tf.norm(u - v, ord=2, axis=2)
    hat = tf.maximum(0., 1. - dist**2)
    return hat


def kernel_layer(x, n_hid=500, n_dim=5, activation=tf.nn.sigmoid, lambda_s=lambda_s,
                 lambda_2=lambda_2, name=''):
    """
    a kernel sparsified layer

    :param x: input [batch, channels]
    :param n_hid: number of hidden units
    :param n_dim: number of dimensions to embed for kernelization
    :param activation: output activation
    :param name: layer name for scoping
    :return: layer output, regularization term
    """

    # define variables
    with tf.variable_scope(name):
        W = tf.get_variable('W', [x.shape[1], n_hid])
        n_in = x.get_shape().as_list()[1]
        u = tf.get_variable('u', initializer=tf.random.truncated_normal([n_in, 1, n_dim], 0., 1e-3))
        v = tf.get_variable('v', initializer=tf.random.truncated_normal([1, n_hid, n_dim], 0., 1e-3))
        b = tf.get_variable('b', [n_hid])

    # compute sparsifying kernel
    # as u and v move further from each other for some given pair of neurons, their connection
    # decreases in strength and eventually goes to zero.
    w_hat = kernel(u, v)

    # compute regularization terms
    sparse_reg = tf.contrib.layers.l2_regularizer(lambda_s)
    sparse_reg_term = tf.contrib.layers.apply_regularization(sparse_reg, [w_hat])

    l2_reg = tf.contrib.layers.l2_regularizer(lambda_2)
    l2_reg_term = tf.contrib.layers.apply_regularization(l2_reg, [W])

    # compute output
    W_eff = W * w_hat
    y = tf.matmul(x, W_eff) + b
    y = activation(y)
    return y, sparse_reg_term + l2_reg_term


# Instantiate network
y = R
reg_losses = None
for i in range(n_layers):
    y, reg_loss = kernel_layer(y, n_hid, name=str(i))
    reg_losses = reg_loss if reg_losses is None else reg_losses + reg_loss
prediction, reg_loss = kernel_layer(y, n_u, activation=tf.identity, name='out')
reg_losses = reg_losses + reg_loss

# Compute loss (symbolic)
diff = tm*(R - prediction)
sqE = tf.nn.l2_loss(diff)
loss = sqE + reg_losses

# Instantiate L-BFGS Optimizer
optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, options={'maxiter': output_every,
                                                                  'disp': verbose_bfgs,
                                                                  'maxcor': 10},
                                                   method='L-BFGS-B')

# Training and validation loop
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(int(n_epoch / output_every)):
        optimizer.minimize(sess, feed_dict={R: tr}) #do maxiter optimization steps
        pre = sess.run(prediction, feed_dict={R: tr}) #predict ratings

        error = (vm * (np.clip(pre, 1., 5.) - vr) ** 2).sum() / vm.sum() #compute validation error
        error_train = (tm * (np.clip(pre, 1., 5.) - tr) ** 2).sum() / tm.sum() #compute train error

        print('.-^-._' * 12)
        print('epoch:', i, 'validation rmse:', np.sqrt(error), 'train rmse:', np.sqrt(error_train))
        print('.-^-._' * 12)

    with open('summary_ml1m.txt', 'a') as file:
        for a in sys.argv[1:]:
            file.write(a + ' ')
        file.write(str(np.sqrt(error)) + ' ' + str(np.sqrt(error_train))
                   + ' ' + str(seed) + '\n')
        file.close()
