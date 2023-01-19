import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.sparse as sp
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

"""Applies model and attention mechannism.
   From the paper:
   Yu Z, Huang F, Zhao X, Xiao W, Zhang W."Predicting drug-disease associations through layer attention graph
   convolutional network." Brief Bioinform. 2021 Jul 20; 22(4): bbaa243.
"""
def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = tf.random_uniform(
        [input_dim, output_dim],
        minval=-init_range,
        maxval=init_range,
        dtype=tf.float32
    )
    return tf.Variable(initial, name=name)


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out*(1./keep_prob)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj_ = sp.coo_matrix(adj)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_nomalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    adj_nomalized = adj_nomalized.tocoo()
    return sparse_to_tuple(adj_nomalized)


def constructNet(drug_lnc_matrix):
    # drug_matrix = np.matrix(np.zeros((drug_lnc_matrix.shape[0], drug_lnc_matrix.shape[0]), dtype=np.int8))
    # lnc_matrix = np.matrix(np.zeros((drug_lnc_matrix.shape[1], drug_lnc_matrix.shape[1]), dtype=np.int8))
    drug_matrix = np.matrix(np.eye(drug_lnc_matrix.shape[0], dtype=np.int8))
    lnc_matrix = np.matrix(np.eye(drug_lnc_matrix.shape[1], dtype=np.int8))

    mat1 = np.hstack((drug_matrix, drug_lnc_matrix))
    mat2 = np.hstack((drug_lnc_matrix.T, lnc_matrix))
    adj = np.vstack((mat1, mat2))
    return adj


def constructHNet(drug_lnc_matrix, drug_matrix, lnc_matrix):
    mat1 = np.hstack((drug_matrix, drug_lnc_matrix))
    mat2 = np.hstack((drug_lnc_matrix.T, lnc_matrix))
    return np.vstack((mat1, mat2))


def comparision(drug_lnc_matrix, train_matrix, random_index2, drug_matrix, lnc_matrix, predict_y):
    train_matrix[tuple(np.array(random_index2).T)] = 1
    train_set = np.mat(np.where(train_matrix.T.reshape(-1) == 1))[1, ].A.reshape(-1)
    train_matrix[tuple(np.array(random_index2).T)] = 0
    test_set = np.mat(np.where(train_matrix.T.reshape(-1) == 0))[1, ].A.reshape(-1)

    feature = pd.DataFrame(drug_lnc_matrix)
    feature = feature.T
    feature = pd.melt(feature.reset_index(), id_vars=['index'], value_vars=feature.columns.values)
    feature.columns = ['lnc', 'drug', 'label']
    feature['index'] = range(feature.shape[0])
    matrix_d = pd.DataFrame(drug_matrix).reset_index()
    matrix_l = pd.DataFrame(lnc_matrix).reset_index()
    matrix_d.rename(columns={'index': 'drug'}, inplace=True)
    matrix_l.rename(columns={'index': 'lnc'}, inplace=True)
    all_pairs = pd.merge(pd.merge(feature, matrix_d, left_on='drug', right_on='drug'),
                         matrix_l, left_on='lnc', right_on='lnc')
    feature = np.array(all_pairs.drop(labels=['drug', 'lnc', 'label', 'index'], axis=1))
    label = np.array(all_pairs['label'])

    train_feature = feature[train_set, ]
    train_label = label[train_set, ]
    test_feature = feature[test_set, ]
    test_label = label[test_set, ]

    print('Training SVM')
    model_svm = SVC(kernel='rbf')
    model_svm.fit(train_feature, train_label)
    predict_svm = model_svm.predict(test_feature)
    print('Training RF')
    rf = RandomForestClassifier()
    param = {"n_estimators": [120, 200, 500], "max_depth": [5, 10, 25, 40]}
    model_rf = GridSearchCV(rf, param_grid=param, cv=2)
    model_rf.fit(train_feature, train_label)
    predict_rf = model_rf.predict(test_feature)
    print('Training ENet')
    enet = ElasticNet()
    param = {"alpha": [0.01, 0.05, 0.1, 0.5, 1, 5, 10], "l1_ratio": [0.01, 0.05, 0.1, 0.5, 1]}
    model_enet = GridSearchCV(enet, param_grid=param, cv=2)
    model_enet.fit(train_feature, train_label)
    predict_enet = model_enet.predict(test_feature)
    print('Training KNN')
    model_knn = neighbors.KNeighborsRegressor(n_neighbors=5, n_jobs=1)
    model_knn.fit(train_feature, train_label)
    predict_knn = model_knn.predict(test_feature)
    predict_gcn = predict_y.T.reshape(-1)[test_set].reshape(-1, 1)
    predict_all = np.hstack((test_label.reshape(-1, 1), predict_svm.reshape(-1, 1), predict_rf.reshape(-1, 1),
                             predict_enet.reshape(-1, 1), predict_knn.reshape(-1, 1), predict_gcn))
    return predict_all
