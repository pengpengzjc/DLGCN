import tensorflow as tf
from utils import *


class GraphConvolution():
    """Graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = tf.nn.dropout(x, 1-self.dropout)
            x = tf.matmul(x, self.vars['weights'])
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
        return outputs


class GraphConvolutionSparse():
    """Graph convolution layer for sparse inputs."""

    def __init__(self, input_dim, output_dim, adj, features_nonzero, name, dropout=0., act=tf.nn.relu):
        self.name = name
        self.vars = {}
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(
                input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
        return outputs


class InnerProductDecoder():
    """Decoder model layer for link prediction, applying Inductuive Matix Completion.
       From the paper:
       Zeng X, Zhu S, Lu W, Liu Z, Huang J, Zhou Y, Fang J, Huang Y, Guo H, Li L, Trapp BD, Nussinov R, Eng C, Loscalzo
       J, Cheng F. Target identification among known drugs by deep learning from heterogeneous networks. Chem Sci. 2020
       Jan 13;11(7):1775-1797.
    """

    def __init__(self, input_dim, name, num_r, dropout=0., act=tf.nn.sigmoid, b=16):
        self.name = name
        self.vars = {}
        self.issparse = False
        self.dropout = dropout
        self.act = act
        self.num_r = num_r
        self.r = b
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights1'] = weight_variable_glorot(
                input_dim, b, name='weights1')
            self.vars['weights2'] = weight_variable_glorot(
                input_dim, b, name='weights2')

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            inputs = tf.nn.dropout(inputs, 1-self.dropout)
            r = inputs[0:self.num_r, :]
            d = inputs[self.num_r:, :]
            r = tf.matmul(r, self.vars['weights1'])
            d = tf.matmul(d, self.vars['weights2'])
            d = tf.transpose(d)
            x = tf.matmul(r, d)
            x = tf.reshape(x, [-1])
            outputs = self.act(x)
        return outputs