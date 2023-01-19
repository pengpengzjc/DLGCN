from clr import cyclic_learning_rate
import tensorflow as tf


class Optimizer():
    """Applies PU-learning.
       From the paper:
       Zeng X, Zhu S, Lu W, Liu Z, Huang J, Zhou Y, Fang J, Huang Y, Guo H, Li L, Trapp BD, Nussinov R, Eng C, Loscalzo
       J, Cheng F. Target identification among known drugs by deep learning from heterogeneous networks. Chem Sci. 2020
       Jan 13;11(7):1775-1797.
    """

    def __init__(self, model, preds, labels, lr, num_u, num_v, pos):
        norm = num_u*num_v / float((num_u*num_v-pos) * 2)
        preds_sub = preds
        labels_sub = labels
        pos_weight = float(num_u*num_v-pos)/pos

        global_step = tf.Variable(0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=cyclic_learning_rate(global_step=global_step, learning_rate=lr*0.1,
                                                                                   max_lr=lr, mode='exp_range', gamma=.995))

        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))

        self.opt_op = self.optimizer.minimize(
            self.cost, global_step=global_step,)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
