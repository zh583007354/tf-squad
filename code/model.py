 # -*- coding:utf-8 -*- 
import numpy as np
import tensorflow as tf
from my.nn import linear, highway_network, softsel, multi_conv1d
from my.general import flatten, reconstruct, add_wd, exp_mask


class Model(object):
    """docstring for Model"""
    def __init__(self, args):
        self.args = args
        self.embeddings = args.embeddings

        self.in_c = tf.placeholder('int32', (None, None), name='c')
        self.in_cc = tf.placeholder('int32', (None, None, None), name='cc')
        self.in_q = tf.placeholder('int32', (None, None), name='q')
        self.in_qc = tf.placeholder('int32', (None, None, None), name='qc')
        self.in_c_mask = tf.placeholder('bool', (None, None), name='cm')
        self.in_q_mask = tf.placeholder('bool', (None, None), name='qm')
        self.in_y1 = tf.placeholder('int32', (None, None), name='y1')
        self.in_y2 = tf.placeholder('int32', (None, None), name='y2')
        self.DP = tf.placeholder('float', [], name='DP')

        self.logits1 = None
        self.logits2 = None
        self.y1_pre = None
        self.y2_pre = None

        self.loss = None

        self._build_forward()
        self._build_loss()


    def _build_forward(self):
        args = self.args

        HS = args.hidden_size
        VC = args.char_voc_size
        dc = args.char_emb_size
        dco = args.char_out_size
        
        with tf.variable_scope("emb"), tf.device("/cpu:0"):
            word_emb_mat = tf.Variable(self.embeddings, name="word_emb_mat", dtype='float', trainable=False)
            with tf.variable_scope("word"):
                input_c = tf.nn.embedding_lookup(word_emb_mat, self.in_c)
                input_q = tf.nn.embedding_lookup(word_emb_mat, self.in_q)


            if args.use_char_emb:
                char_emb_mat = tf.get_variable("char_emb_mat", shape=[VC, dc], dtype='float')
                with tf.variable_scope("char"):
                    Acc = tf.nn.embedding_lookup(char_emb_mat, self.in_cc)
                    Acq = tf.nn.embedding_lookup(char_emb_mat, self.in_qc)

                    filter_sizes = list(map(int, args.out_channel_dims.split(',')))
                    heights = list(map(int, args.filter_heights.split(',')))
                    assert sum(filter_sizes) == dco, (filter_sizes, dco)
                    with tf.variable_scope("conv"):
                        cc = multi_conv1d(Acc, filter_sizes, heights, "VALID", keep_prob=self.DP, scope="cc")

                        if args.share_cnn_weights:
                            tf.get_variable_scope().reuse_variables()
                            qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", keep_prob=self.DP, scope="cc")
                        else:
                            qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", keep_prob=self.DP, scope="qq")

                input_c = tf.concat([cc, input_c], axis=2)
                input_q = tf.concat([qq, input_q], axis=2)

        if args.highway:
            with tf.variable_scope("highway"):
                input_c = highway_network(input_c, args.highway_num_layers, True)#, input_keep_prob=self.DP
                tf.get_variable_scope().reuse_variables()
                input_q = highway_network(input_q, args.highway_num_layers, True)#, input_keep_prob=self.DP

        cell = tf.contrib.rnn.GRUCell(HS)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.DP)#, output_keep_prob=self.DP

        c_len = tf.reduce_sum(tf.cast(self.in_c_mask, 'int32'), 1)  # [N]
        q_len = tf.reduce_sum(tf.cast(self.in_q_mask, 'int32'), 1)  # [N]
        
        with tf.variable_scope("prepro"):
            (fw_c, bw_c), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,
                                                              cell_bw=cell,
                                                              inputs=input_c,
                                                              sequence_length=c_len,
                                                              dtype='float',
                                                              time_major=False,
                                                              scope='c')
            (fw_q, bw_q), (fw_f_q, bw_f_q) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,
                                                              cell_bw=cell,
                                                              inputs=input_q,
                                                              sequence_length=q_len,
                                                              dtype='float',
                                                              time_major=False,
                                                              scope='q')

            out_c = tf.concat([fw_c, bw_c], 2)
            out_f_q = tf.concat([fw_f_q, bw_f_q], 1)
            out_q = tf.concat([fw_q, bw_q], 2)

        
        g0_cell = tf.contrib.rnn.GRUCell(HS)
        g0_cell = tf.contrib.rnn.DropoutWrapper(g0_cell, input_keep_prob=self.DP)#, output_keep_prob=self.DP
        
        g1_cell = tf.contrib.rnn.GRUCell(HS)
        g1_cell = tf.contrib.rnn.DropoutWrapper(g1_cell, input_keep_prob=self.DP)#, output_keep_prob=self.DP
        
        g2_cell = tf.contrib.rnn.GRUCell(HS)
        g2_cell = tf.contrib.rnn.DropoutWrapper(g2_cell, input_keep_prob=self.DP)#, output_keep_prob=self.DP


        with tf.variable_scope("main"):

            if args.att_func == 'bilinear':
                logits1 = bilinear_attention(out_c, out_f_q, W_bilinear1, self.in_c_mask)
                logits2 = bilinear_attention(out_c, out_f_q, W_bilinear2, self.in_c_mask)
            elif args.att_func == 'aoa':
                #
                logits1 = attention_over_attention(out_c, out_q, self.in_c_mask, self.in_q_mask)
                logits2 = attention_over_attention(out_c, out_q, self.in_c_mask, self.in_q_mask)
                #
            elif args.att_func == 'biDAF':
                c0 = bidir_attention(args, out_c, out_q, self.in_c_mask, self.in_q_mask, scope="c0")
                (fw_g0, bw_g0), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=g0_cell,
                                                                    cell_bw=g0_cell,
                                                                    inputs=c0,
                                                                    sequence_length=c_len,
                                                                    dtype='float',
                                                                    time_major=False,
                                                                    scope='g0')
                g0 = tf.concat([fw_g0, bw_g0], 2)

                (fw_g1, bw_g1), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=g1_cell,
                                                                    cell_bw=g1_cell,
                                                                    inputs=g0,
                                                                    sequence_length=c_len,
                                                                    dtype='float',
                                                                    time_major=False,
                                                                    scope='g1')
                g1 = tf.concat([fw_g1, bw_g1], 2) #[batch * c_len * 2HS]

                logits1 = linear([g1, c0], 1, True, scope='logits1', squeeze=True, input_keep_prob=self.DP)#, name_w='logits1_w', name_b='logits1_b'
                if self.in_c_mask is not None:
                    logits1 = exp_mask(logits1, self.in_c_mask)
                
                # I don't konw why named this
                # a = softmax(logits1)
                # ali = sum(a*g1)
                ali = softsel(g1, logits1) #[batch * 2HS]
                ali = tf.tile(tf.expand_dims(ali, 1), [1, tf.shape(self.in_c)[1], 1])

                (fw_g2, bw_g2), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=g2_cell,
                                                                    cell_bw=g2_cell,
                                                                    inputs=tf.concat([c0, g1, ali, g1*ali], 2),
                                                                    sequence_length=c_len,
                                                                    dtype='float',
                                                                    time_major=False,
                                                                    scope='g2')
                g2 = tf.concat([fw_g2, bw_g2], 2)
                logits2 = linear([g2, c0], 1, True, scope='logits2', squeeze=True, input_keep_prob=self.DP)#, name_w='logits2_w', name_b='logits2_b'
                if self.in_c_mask is not None:
                    logits2 = exp_mask(logits2, self.in_c_mask)


            else:
                raise NotImplementedError('att_func = %s' % args.att_func)

            
            y1_pre = tf.argmax(logits1, 1)
            y2_pre = tf.argmax(logits2, 1)

            self.logits1 = logits1
            self.logits2 = logits2
            self.y1_pre = y1_pre
            self.y2_pre = y2_pre
    def _build_loss(self):
        args = self.args

        loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits1, labels=self.in_y1))
        #tf.add_to_collection('',loss1)
        loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits2, labels=self.in_y2))
        #tf.add_to_collection
        self.loss = tf.add_n([loss1, loss2], name="loss")

    def get_loss(self):
        return self.loss

    # def get_feed_dict():
def bilinear_attention(c, q, w, c_mask):    
    qW = tf.matmul(q, w)
    #qWp = tf.multiply(out_c, tf.expand_dims(qW, 1))
    qWc = tf.reduce_sum((c * tf.expand_dims(qW, 1)), 2)
    alphas = tf.nn.softmax(qWc)
    alphas = alphas * tf.cast(c_mask, 'float')
    alphas = alphas / tf.expand_dims(tf.reduce_sum(alphas, 1), 1)
    # self.alphas = alphas
    return alphas

def attention_over_attention(c, q, c_mask, q_mask):
    W = tf.matmul(c, tf.transpose(q, [0, 2, 1]))
    #Column_wise_softmax 
    c_w_s = tf.nn.softmax(W, axis=1)
    c_w_s = c_w_s * tf.cast(c_mask, 'float')
    c_w_s = c_w_s / tf.expand_dims(tf.reduce_sum(c_w_s, 1), 1)
    #Row_wise_softmax 
    r_w_s = tf.nn.softmax(W, axis=2)
    r_w_s = r_w_s * tf.cast(q_mask, 'float')
    r_w_s = r_w_s / tf.expand_dims(tf.reduce_sum(r_w_s, 2), 2)
    #Row_wise_Average 
    r_w_a = tf.reduce_mean(r_w_s, 1)
    alphas = tf.reduce_sum((c_w_s * tf.expand_dims(r_w_a, 1)), 2)
    return alphas

def bidir_attention(args, c, q, c_mask=None, q_mask=None, scope=None):
    with tf.variable_scope(scope or "bidir_attention"):
        len_c = tf.shape(c)[1]
        len_q = tf.shape(q)[1]
        c_aug = tf.tile(tf.expand_dims(c, 2), [1, 1, len_q, 1])
        q_aug = tf.tile(tf.expand_dims(q, 1), [1, len_c, 1, 1])
        if c_mask is None:
            cq_mask = None
        else:
            c_mask_aug = tf.tile(tf.expand_dims(c_mask, 2), [1, 1, len_q])
            q_mask_aug = tf.tile(tf.expand_dims(q_mask, 1), [1, len_c, 1])
            cq_mask = c_mask_aug & q_mask_aug
        # S = alpha(C;Q)
        # alpha = w(c;q;c*q)  
        logits = linear([c_aug, q_aug, c_aug*q_aug], 1, True, scope='bidir_logits', squeeze=True)#, name_w='bidir_att_w', name_b='bidir_att_b'
        if c_mask is not None:
            logits = exp_mask(logits, cq_mask)
        # a[i] = softmax(S[i])
        # ~q = sum(a*q)
        # ~q[i] = sum(a[ij]*q[j])
        q_a = softsel(q_aug, logits)
        # b = softmax(maxcol(S))
        # ~c = sum(b*c) = sum(b[i]*c[i])
        c_a = softsel(c, tf.reduce_max(logits, 2))
        # ~h is tiled T times across the column
        c_a = tf.tile(tf.expand_dims(c_a, 1), [1, len_c, 1])

        # g = beta(h, ~u, ~h) = [h;~u;h*~u;h*~h]
        g = tf.concat([c, q_a, c * q_a, c * c_a], 2)
        
        return g





