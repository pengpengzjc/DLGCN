import math
import numpy as np
from sklearn.metrics import roc_auc_score


def my_rand_walk(w, p0, r):
    pt = p0
    for k in range(1000):
        pt1 = (1 - r) * np.matmul(w, pt) + r * p0
        pt = pt1
    return pt


def norm(adj):
    degree = np.array(adj.sum(1)).reshape(-1)
    degree = np.power(degree, -0.5)
    degree = np.diag(degree)
    return degree.dot(adj).dot(degree)


if __name__ == "__main__":
    """Code for TL-HGBI.
       From the paper:
       Wang W, Yang S, Zhang X, Li J. Drug repositioning by integrating target information through a heterogeneous 
       network model. Bioinformatics. 2014 Oct 15;30(20):2923-30.
    """
    drug_sim = np.loadtxt(r'.../drug_sim.txt', delimiter='\t')
    drug_sim = drug_sim - np.eye(drug_sim.shape[0])
    a = math.ceil(drug_sim.shape[0] * drug_sim.shape[0] * 0.85)
    b = np.partition(drug_sim.reshape(-1), -a)[-a]
    drug_sim = np.where(drug_sim >= b, 1, 0)
    drug_sim = drug_sim + np.eye(drug_sim.shape[0])
    lnc_sim = np.loadtxt(r'.../lnc_sim.txt', delimiter='\t')
    lnc_sim = lnc_sim - np.eye(lnc_sim.shape[0])
    a = math.ceil(lnc_sim.shape[0] * lnc_sim.shape[0] * 0.85)
    b = np.partition(lnc_sim.reshape(-1), -a)[-a]
    lnc_sim = np.where(lnc_sim >= b, 1, 0)
    lnc_sim = lnc_sim + np.eye(lnc_sim.shape[0])
    drug_lnc_matrix = np.loadtxt(r'.../matrix_train.txt', delimiter='\t')

    w1 = np.array(drug_sim)
    w2 = np.array(lnc_sim)
    p = np.array(drug_lnc_matrix)
    p = p.reshape((-1, 1), order='F')
    C = np.matrix(np.zeros((p.shape[0] * p.shape[1], p.shape[0] * p.shape[1]), dtype=np.int8))
    C1 = C

    n = 0
    for i in range(drug_lnc_matrix.shape[0]):
        for j in range(drug_lnc_matrix.shape[1]):
            c1 = np.outer(w1[i, :], w2[:, j])
            C[n, :] = c1.reshape((1, -1), order='F')
            n += 1

    C2 = norm(C)
    p2 = np.ones(p.shape)
    p2 = p2 * np.sum(p) / np.sum(p2)
    a = 0.8
    pf = my_rand_walk(C2, p2, a)
    result = np.hstack([p, pf])
    roc_auc = roc_auc_score(p, pf)
    print(roc_auc)
    np.savetxt(r'.../matrix_train_result.txt', result, delimiter='\t')
