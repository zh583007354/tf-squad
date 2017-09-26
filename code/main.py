 # -*- coding:utf-8 -*- 
import numpy as np
import tensorflow as tf
import os
import sys
import time
import utils
from model import Model 
from trainer import Trainer
from evaluator import Evaluator
from gen_answer_file import *

def main(args):
    with tf.device("/gpu:1"):
        print('-' * 50)
        print('Load data files..')
        if args.debug:
            print('*' * 10 + ' Train')
            train_data = utils.load_data(args, 'train', 100)
            print('*' * 10 + ' Test')
            test_data = utils.load_data(args, 'test', 100)
            print('*' * 10 + ' Dev')
            dev_data = utils.load_data(args, 'dev', 100)
        else:
            print('*' * 10 + ' Train')
            train_data = utils.load_data(args, 'train')
            print('*' * 10 + ' Test')
            test_data = utils.load_data(args, 'test')
            print('*' * 10 + ' Dev')
            dev_data = utils.load_data(args, 'dev')

        print('-' * 50)
        print('Build dictionary..')
        args.word_dict, args.char_dict = utils.build_dict(train_data.data[0]+train_data.data[1]+test_data.data[0]+test_data.data[1]+dev_data.data[0]+dev_data.data[1])
        print('-' * 50)
        # Load embedding file
        args.embeddings = utils.gen_embeddings(args.word_dict, args.embedding_size, args.embedding_file)
        (args.word_voc_size, args.embedding_size) = args.embeddings.shape
        args.char_voc_size = len(args.char_dict)
        train_data.vectorize(args.word_dict, args.char_dict)
        test_data.vectorize(args.word_dict, args.char_dict)

        
        config_gpu = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config_gpu.gpu_options.allow_growth = True
        with tf.Session(config=config_gpu) as sess:
            
            model = Model(args) 
            trainer = Trainer(args, model)
            evaluator = Evaluator(args, model)
            
            tf.global_variables_initializer().run()
            
            timestamp = str(int(time.time()))
            out_dir = os.path.join(args.out_dir, timestamp)
            checkpoint_dir = os.path.join(out_dir, "checkpoints")
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)        
            saver = tf.train.Saver(tf.global_variables())
            
            if args.load:
                cpkl = os.path.join(args.out_dir, "1506152182/checkpoints/model-17000")
                saver.restore(sess, cpkl)
                test_acc = evaluator.get_evaluation(sess, test_data.gen_minbatches(args.batch_size))
                print('Test accuracy: %.2f %%' % test_acc)

                dev_data.vectorize(args.word_dict, args.char_dict)
                answers = evaluator.get_answers(sess, dev_data.gen_minbatches(args.batch_size))
                gen_answer_file(answers)
                # return

            # Training
            print('-' * 50)
            print('Start training..')
            best_acc = 0
            start_time = time.time()
            last_time = start_time        
            n_updates = 0
            batch100_time = 0

            for epoch in range(args.num_epoches):
                for idx, batch in enumerate(train_data.gen_minbatches(args.batch_size, shuffle=True)):
                    
                    train_loss, train_op = trainer.step(sess, batch)
                    batch_time = time.time() - last_time
                    if idx % 20 == 0:
                        print('Epoch = %d, iter = %d, loss = %.2f, batch time = %.2f (s)' %
                                 (epoch, idx, train_loss, batch_time))
                    
                    n_updates += 1
                    batch100_time = batch100_time + batch_time
                    # Evalution
                    if n_updates % args.eval_iter == 0:
                        print('time pre 100 batches: %.2f (s)' % (batch100_time))
                        batch100_time = 0
                        start_examples = np.random.randint(0, train_data.num_examples - test_data.num_examples)
                        end_examples = start_examples + test_data.num_examples
                        train_acc = evaluator.get_evaluation(sess, train_data.gen_minbatches(args.batch_size, start_examples, end_examples))
                        print('Epoch = %d, iter = %d, train_acc = %.2f %%' % (epoch, idx, train_acc))

                        test_acc = evaluator.get_evaluation(sess, test_data.gen_minbatches(args.batch_size))
                        print('Epoch = %d, iter = %d, test_acc = %.2f %%, Best test accuracy: %.2f %%' % (epoch, idx, test_acc, best_acc))

                        if test_acc > best_acc:
                            best_acc = test_acc
                            print('Best test accuracy: epoch = %d, n_udpates = %d, acc = %.2f %%' % (epoch, n_updates, test_acc))
                            path = saver.save(sess, checkpoint_prefix, global_step=n_updates)
                            print("Saved model checkpoint to {}\n".format(path))
                    last_time = time.time()