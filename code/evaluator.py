 # -*- coding:utf-8 -*- 
import numpy as np
import tensorflow as tf

class Evaluator(object):
    """docstring for Evaluator"""
    def __init__(self, args, model):
        self.args = args
        self.model = model
        
    def get_evaluation(self, sess, batches):
        """
            Evaluate accuracy on `all_examples`.
        """
        acc = 0
        n_examples = 0
        for idx, batch in enumerate(batches):
            feed = {self.model.in_c: batch[0],
                    self.model.in_cc: batch[1],
                    self.model.in_c_mask: batch[2],
                    self.model.in_q: batch[3],
                    self.model.in_qc: batch[4],
                    self.model.in_q_mask: batch[5],
                    self.model.in_y1: batch[6],
                    self.model.in_y2: batch[7],
                    self.model.DP: 1.0}

            y1_pre, y2_pre = sess.run([self.model.y1_pre, self.model.y2_pre], feed)
            acc += np.sum((np.equal(y1_pre, np.argmax(batch[6], 1)) & np.equal(y2_pre, np.argmax(batch[7], 1))).astype('int'))
            n_examples += len(batch[0])
        return acc * 100.0 / n_examples

    def get_answers(self, sess, batches):
        answers = []
        for idx, batch in enumerate(batches):
            feed = {self.model.in_c: batch[0],
                    self.model.in_cc: batch[1],
                    self.model.in_c_mask: batch[2],
                    self.model.in_q: batch[3],
                    self.model.in_qc: batch[4],
                    self.model.in_q_mask: batch[5],
                    self.model.DP: 1.0}
            y1_pre, y2_pre = sess.run([self.model.y1_pre, self.model.y2_pre], feed)
            qid = batch[8]
            for i in range(len(qid)):
                answers.append([qid[i], y1_pre[i], y2_pre[i]])
            print('Have generated %d pairs answers', self.args.batch_size * idx)
        return answers