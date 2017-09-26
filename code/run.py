 # -*- coding:utf-8 -*- 
import os
import tensorflow as tf
from main import main as m


flags = tf.app.flags

flags.DEFINE_string("data_dir", "../data/squad/my", "data_dir.")
flags.DEFINE_string("out_dir", "out", "out_dir.")
flags.DEFINE_string("embedding_file", "../data/squad/my/glove.6B.100d.txt", "Word embedding file.")

flags.DEFINE_integer("hidden_size", 128, "Hidden size of RNN units.")
flags.DEFINE_integer("char_out_size", 100, "char-level word embedding size [100]")
flags.DEFINE_integer("char_emb_size", 8, "Char emb size [8]")
flags.DEFINE_string("out_channel_dims", "22,24,26,28", "Out channel dims of Char-CNN, separated by commas [100]")
flags.DEFINE_string("filter_heights", "2,3,4,5", "Filter heights of Char-CNN, separated by commas [5]")
flags.DEFINE_bool("highway", True, "Use highway? [True]")
flags.DEFINE_integer("highway_num_layers", 2, "highway num layers [2]")
flags.DEFINE_bool("share_cnn_weights", True, "Share Char-CNN weights [True]")
flags.DEFINE_bool("share_lstm_weights", True, "Share pre-processing (phrase-level) LSTM weights [True]")
flags.DEFINE_string("att_func", "biDAF", "Attention function: biDAF or ...")

flags.DEFINE_integer("batch_size", 60, "Batch size.")
flags.DEFINE_integer("embedding_size", 100, "Embedding size.")
flags.DEFINE_integer("num_epoches", 100, "Number of epoches.")
flags.DEFINE_integer("eval_iter", 100, "Evaluation on dev set after K updates.")

flags.DEFINE_float("dropout_rate", 0.8, "Dropout rate.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for Adam")

flags.DEFINE_bool("use_char_emb", True, "use char emb? [True]")
flags.DEFINE_boolean("debug", False, "Debugging mode? [False]")
flags.DEFINE_boolean("load", False, "load: load last train results and generate the answer file.")



def main(_):
    args = flags.FLAGS

    print("\nParameters:")
    for attr, value in sorted(args.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")
    m(args)

if __name__ == "__main__":
    tf.app.run()
