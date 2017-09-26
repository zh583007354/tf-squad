#-*- coding:utf-8 -*-

import tensorflow as tf


class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.opt = tf.train.AdamOptimizer(args.learning_rate)
        self.loss = model.get_loss()
        self.train_op = self.opt.minimize(self.loss)
    
    def get_train_op(self):
        return self.train_op

    def step(self, sess, batch):
        assert isinstance(sess, tf.Session)
        feed = {self.model.in_c: batch[0],
                self.model.in_cc: batch[1],
                self.model.in_c_mask: batch[2],
                self.model.in_q: batch[3],
                self.model.in_qc: batch[4],
                self.model.in_q_mask: batch[5],
                self.model.in_y1: batch[6],
                self.model.in_y2: batch[7],
                self.model.DP: 0.8}
        loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed)
        return loss, train_op
