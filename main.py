import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
import gc
import math
import random
from clac_metric import cv_model_evaluate, get_metrics
from utils import *
from model import GCNModel
from opt import Optimizer


def PredictScore(train_drug_lnc_matrix, drug_matrix, lnc_matrix, seed, epochs, emb_dim, dp, lr,  adjdp):
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    adj = constructHNet(train_drug_lnc_matrix, drug_matrix, lnc_matrix)
    adj = sp.csr_matrix(adj)
    association_nam = train_drug_lnc_matrix.sum()
    X = constructNet(train_drug_lnc_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_drug_lnc_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))

    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'adjdp': tf.placeholder_with_default(0., shape=())
    }
    model = GCNModel(placeholders, num_features, emb_dim,
                     features_nonzero, adj_nonzero, train_drug_lnc_matrix.shape[0], name='LAGCN')
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr, num_u=train_drug_lnc_matrix.shape[0], num_v=train_drug_lnc_matrix.shape[1], association_nam=association_nam)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        if epoch % 500 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost))
    print('Optimization Finished!')
    # print(sess.run(tf.get_default_graph().get_tensor_by_name('LAGCN/gcn_sparse_layer_vars/weights:0')))
    # print(sess.run('LAGCN/gcn_sparse_layer_vars/weights:0'))
    # print(sess.run('LAGCN/gcn_dense_layer_vars/weights:0'))
    # print(sess.run('LAGCN/gcn_dense_layer2_vars/weights:0'))
    # print(sess.run('LAGCN/gcn_decoder_vars/weights:0'))
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    emb = sess.run(model.embeddings, feed_dict=feed_dict)
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    sess.close()
    return res


def cross_validation_experiment(drug_lnc_matrix, drug_matrix, lnc_matrix, seed, epochs, emb_dim, dp, lr, adjdp):
    index_matrix1 = np.mat(np.where(drug_lnc_matrix == 1))
    index_matrix2 = np.mat(np.where(drug_lnc_matrix == 0))
    index_matrix = np.hstack([index_matrix1, index_matrix2])
    random_index = index_matrix.T.tolist()
    association_nam = index_matrix.shape[1]
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = association_nam
    cv_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam % k_folds]).reshape(k_folds, cv_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + random_index[association_nam - association_nam % k_folds:]
    random_index = temp

    train_result = np.empty(shape=[0, 2])
    print("evaluating drug-lncrna")
    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k+1))
        train_matrix = np.matrix(drug_lnc_matrix, copy=True)
        train_matrix[tuple(np.array(random_index[k]).T)] = 0
        train_mask = np.mat(np.ones(shape=drug_lnc_matrix.shape))
        train_mask[tuple(np.array(random_index[k]).T)] = 0
        drug_len = drug_lnc_matrix.shape[0]
        lnc_len = drug_lnc_matrix.shape[1]
        drug_lncrna_res = PredictScore(train_matrix, drug_matrix, lnc_matrix, seed, epochs, emb_dim, dp, lr,  adjdp)
        predict_y_proba = drug_lncrna_res.reshape(drug_len, lnc_len)

        fold_result = np.hstack([drug_lnc_matrix[tuple(np.array(random_index[k]).T)].reshape(-1, 1),
                                 predict_y_proba[tuple(np.array(random_index[k]).T)].reshape(-1, 1)])
        train_result = np.vstack([train_result, fold_result])
        del train_matrix
        gc.collect()
    return train_result


if __name__ == "__main__":
    drug_sim = np.loadtxt(r'.../drug_sim.txt', delimiter='\t')
    drug_sim = drug_sim - np.eye(drug_sim.shape[0])
    a = math.ceil(drug_sim.shape[0] * drug_sim.shape[0] * 0.05)
    b = np.partition(drug_sim.reshape(-1), -a)[-a]
    drug_sim = np.where(drug_sim >= b, 1, 0)
    drug_sim = drug_sim + np.eye(drug_sim.shape[0])
    lnc_sim = np.loadtxt(r'.../lnc_sim.txt', delimiter='\t')
    lnc_sim = lnc_sim - np.eye(lnc_sim.shape[0])
    a = math.ceil(lnc_sim.shape[0] * lnc_sim.shape[0] * 0.05)
    b = np.partition(lnc_sim.reshape(-1), -a)[-a]
    lnc_sim = np.where(lnc_sim >= b, 1, 0)
    lnc_sim = lnc_sim + np.eye(lnc_sim.shape[0])
    drug_lnc_matrix = np.loadtxt(r'.../matrix_train.txt', delimiter='\t')

    epoch = 4000
    emb_dim = 64
    lr = 0.001
    adjdp = 0.1
    dp = 0.1
    abc = 1.5
    predict_result = np.empty(shape=[0, 2])

    circle_time = 1
    if circle_time >= 1:
        for i in range(circle_time):
            result= cross_validation_experiment(
                drug_lnc_matrix, drug_sim * abc, lnc_sim * abc, i, epoch, emb_dim, dp, lr, adjdp)
            predict_result = np.vstack([predict_result, result])
        np.savetxt(r'C:/Users/朱济村/Desktop/predict_score.txt', predict_result, delimiter='\t')

    drug_lncrna_m = PredictScore(drug_lnc_matrix, drug_sim * abc, lnc_sim * abc, 1, epoch, emb_dim, dp, lr, adjdp)
    predict_y_proba = drug_lncrna_m.reshape(drug_lnc_matrix.shape[0], drug_lnc_matrix.shape[1])
    predict_metric = get_metrics(drug_lnc_matrix.reshape(-1), drug_lncrna_m)
    # print("predict metric:", predict_metric)
    np.savetxt(r'C:/Users/朱济村/Desktop/predict_result.txt', predict_y_proba, delimiter='\t')

