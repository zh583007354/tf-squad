# -*- coding:utf-8 -*- 
import os
import tensorflow as tf
import numpy as np
import pickle
import gzip
from collections import Counter

class DataSet(object):
    """docstring for DataSet"""
    def __init__(self, data, data_type):
        self.data = data
        self.data_type = data_type
        # data[0] = documents
        # data[1] = questions
        # data[2] = answers or ids
        self.num_examples = len(self.data[0])
        self.in_x1 = None
        self.in_xc1 = None
        self.in_x2 = None
        self.in_xc2 = None
        self.in_y1 = None
        self.in_y2 = None


    def vectorize(self, word_dict, char_dict, sort_by_len=True, verbose=True):
        
        in_x1, in_xc1, in_x2, in_xc2, in_y1, in_y2, in_qid = [], [], [], [], [], [], []

        for idx, (d, q, a) in enumerate(zip(self.data[0], self.data[1], self.data[2])):
            d_words = d.split(' ')
            q_words = q.split(' ') 
            seq1, seq2, seq3 = [], [], []
            char1, char2 = [], []
            for w in d_words:
                if w in word_dict:
                    seq1.append(word_dict[w])
                else:
                    seq1.append(1)
                chars = []
                for c in w:
                    # if c in char_dict:
                    chars.append(char_dict[c])
                char1.append(chars)
            for w in q_words:
                if w in word_dict:
                    seq2.append(word_dict[w])    
                else:
                    seq2.append(1)
                chars = []
                for c in w:
                    # if c in char_dict:
                    chars.append(char_dict[c])
                char2.append(chars)
            # seq1 = [word_dict[w] if w in word_dict else 0 for w in d_words]
            # #列表中放的是句子中每个词的索引(在字典中的位置,是个整数值),不放词本身
            # seq2 = [word_dict[w] if w in word_dict else 0 for w in q_words]
            if self.data_type == 'dev':
                qid = a
            else:
                a_words = a.split(' ')
                seq3 = [int(w) for w in a_words]
            if (len(seq1) > 0) and (len(seq2) > 0):
                in_x1.append(seq1)
                in_xc1.append(char1)
                in_x2.append(seq2)
                in_xc2.append(char2)
                if self.data_type == 'dev':
                    in_qid.append(qid)
                else:
                    in_y1.append(seq3[0])
                    in_y2.append(seq3[-1])
            if verbose and (idx % 10000 == 0):
                print('Vectorization: processed %d / %d' % (idx, self.num_examples))

        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]), reverse=True)

        if sort_by_len:
            # sort by the document length
            sorted_index = len_argsort(in_x1)
            in_x1 = [in_x1[i] for i in sorted_index]
            in_xc1 = [in_xc1[i] for i in sorted_index]
            in_x2 = [in_x2[i] for i in sorted_index]
            in_xc2 = [in_xc2[i] for i in sorted_index]
            if self.data_type == 'dev':
                in_qid = [in_qid[i] for i in sorted_index]
            else:
                in_y1 = [in_y1[i] for i in sorted_index]
                in_y2 = [in_y2[i] for i in sorted_index]
        self.in_x1 = in_x1
        self.in_xc1 = in_xc1
        self.in_x2 = in_x2
        self.in_xc2 = in_xc2
        self.in_y1 = in_y1
        self.in_y2 = in_y2
        self.in_qid = in_qid

        
    def gen_minbatches(self, batch_size, start_examples=None, end_examples=None, shuffle=False):
        """
            Divide examples into batches of size `batch_size`.
        """
        m = 0
        n = 0
        if start_examples is None:
            m = 0
            n = self.num_examples
        else:
            m = start_examples
            n = end_examples

        idx_list = np.arange(m, n, batch_size)
        if shuffle:
            np.random.shuffle(idx_list)
        minibatches = []
        for idx in idx_list:
            minibatches.append(np.arange(idx, min(idx + batch_size, n)))

        for minibatch in minibatches:
            mb_x1 = [self.in_x1[t] for t in minibatch]
            mb_xc1 = [self.in_xc1[t] for t in minibatch]
            mb_x2 = [self.in_x2[t] for t in minibatch]
            mb_xc2 = [self.in_xc2[t] for t in minibatch]
            mb_y1, mb_y2, mb_qid = [], [], []
            if self.data_type == 'dev':
                mb_qid = [self.in_qid[t] for t in minibatch]
                mb_x1, mb_xc1, mb_mask1, _, _ = prepare_data(mb_x1, mb_xc1)
                mb_x2, mb_xc2, mb_mask2, _, _ = prepare_data(mb_x2, mb_xc2)
            else:
                mb_y1 = [self.in_y1[t] for t in minibatch]
                mb_y2 = [self.in_y2[t] for t in minibatch]
                mb_x1, mb_xc1, mb_mask1, mb_y1, mb_y2= prepare_data(mb_x1, mb_xc1, mb_y1, mb_y2)
                mb_x2, mb_xc2, mb_mask2, _, _ = prepare_data(mb_x2, mb_xc2)

            yield (mb_x1, mb_xc1, mb_mask1, mb_x2, mb_xc2, mb_mask2, mb_y1, mb_y2, mb_qid)

def load_data(args, data_type=None, max_example=None):
    """
        load SQuAD data from {train | dev | test}.txt
    """
    data_path = os.path.join(args.data_dir, "{}.txt".format(data_type))
    documents = []
    questions = []
    answers = []
    num_examples = 0
    f = open(data_path, 'r', encoding='utf-8')
    while True:
        line = f.readline()
        if not line:
            break
        question = line.strip().lower()
        answer = f.readline().strip().lower()
        document = f.readline().strip().lower()

        questions.append(question)
        answers.append(answer)
        documents.append(document)
        num_examples += 1

        f.readline()
        if (max_example is not None) and (num_examples >= max_example):
            break
    f.close()
    print('#Examples: %d' % len(documents))

    dataset = DataSet((documents, questions, answers), data_type)
    return dataset


def build_dict(sentences, max_words=100000):
    """
        Build a dictionary for the words in `sentences`.
        Only the max_words ones are kept and the remaining will be mapped to <UNK>.
    """
    word_count = Counter()
    char_count = Counter()
    for sent in sentences:
        for w in sent.split(' '):
            word_count[w] += 1
            for c in w:
                char_count[c] += 1

    ls = word_count.most_common(max_words)
    print('#Words: %d -> %d' % (len(word_count), len(ls)))
    for key in ls[:5]:
        print(key)
    print('...')
    for key in ls[-5:]:
        print(key)

    # leave 0 to NULL
    # leave 1 to UNK
    word_dict = {w[0]: index + 2 for (index, w) in enumerate(ls)}
    word_dict["NULL"] = 0
    word_dict["UNK"] = 1
    char_dict = {c: index + 1 for (index, c) in enumerate(list(char_count))}
    char_dict["NULL"] = 0
    return word_dict, char_dict

def gen_embeddings(word_dict, dim, in_file=None):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
        50000 * 100
        以词在词典中的序号为索引。
    """

    num_words = max(word_dict.values()) + 1
    embeddings = np.random.normal(0, 1, size=[num_words, dim])
    print('Embeddings: %d x %d' % (num_words, dim))

    if in_file is not None:
        print('Loading embedding file: %s' % in_file)
        pre_trained = 0
        for line in open(in_file, 'r', encoding='UTF-8').readlines():
            sp = line.split()
            assert len(sp) == dim + 1
            if sp[0] in word_dict:
                pre_trained += 1
                embeddings[word_dict[sp[0]]] = [float(x) for x in sp[1:]]
        print('Pre-trained: %d (%.2f%%)' %
                     (pre_trained, pre_trained * 100.0 / num_words))
    return embeddings

def prepare_data(seqs, seqs_char, y1=None, y2=None):  
    lengths_sent = [len(seq) for seq in seqs]
    lengths_word, max_len_char = [], []
    for seq_char in seqs_char:
        length_word = []
        for chars in seq_char:
            length_word.append(len(chars))
        lengths_word.append(length_word)
        max_len_char.append(np.max(length_word))
    max_len_word = np.max(max_len_char)
    n_samples = len(seqs)
    max_len_sent = np.max(lengths_sent)
    x = np.zeros((n_samples, max_len_sent)).astype('int32')
    xc = np.zeros((n_samples, max_len_sent, max_len_word)).astype('int32')
    x_mask = np.zeros((n_samples, max_len_sent)).astype('float')
    y1_onehot = np.zeros((n_samples, max_len_sent)).astype('float')
    y2_onehot = np.zeros((n_samples, max_len_sent)).astype('float')

    for idx, seq in enumerate(seqs):
        x[idx, :lengths_sent[idx]] = seq
        x_mask[idx, :lengths_sent[idx]] = 1.0
        if y1 is not None:
            y1_onehot[idx, y1[idx]-1] = 1.0
            y2_onehot[idx, y2[idx]-1] = 1.0
    for idx, seq_char in enumerate(seqs_char):
        for idy, chars in enumerate(seq_char):
            xc[idx][idy][:lengths_word[idx][idy]] = chars
    return x, xc, x_mask, y1_onehot, y2_onehot




