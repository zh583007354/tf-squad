import argparse
import json
import os
import re
import numpy as np
from tqdm import tqdm
import nltk

from helper import *

def main():
    args = get_args()
    prepro(args)


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("E:/Worksation/tf-squad-master")
    source_dir = os.path.join(home, "data", "squad")
    target_dir = "data/squad"
    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument("--train_name", default='train-v1.1.json')
    parser.add_argument('-d', "--debug", action='store_true')
    parser.add_argument("--train_ratio", default=0.9, type=int)
    parser.add_argument("--mode", default="full", type=str)
    parser.add_argument("--single_path", default="", type=str)
    parser.add_argument("--tokenizer", default="PTB", type=str)
    parser.add_argument("--suffix", default="")
    # TODO : put more args here
    return parser.parse_args()


def prepro(args):
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    prepro_each(args, 'train', 0.0, args.train_ratio, out_name='train')
    prepro_each(args, 'train', args.train_ratio, 1.0, out_name='test')
    prepro_each(args, 'dev', out_name='dev')



def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    if args.tokenizer == "PTB":
        def word_tokenize(tokens):
            return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
            #.replace("''", '"').replace("``", '"')
    #train-v1.1.json and dev-1.1.json
    source_path = in_path or os.path.join(args.source_dir, "{}-{}v1.1.json".format(data_type, args.suffix))
    #open
    source_data = json.load(open(source_path, 'r', encoding='utf-8')) 
    fpw = open("data/squad/"+out_name+".txt", 'w', encoding='utf-8')

    # a = article
    start_ai = int(round(len(source_data['data']) * start_ratio))
    stop_ai = int(round(len(source_data['data']) * stop_ratio))

    # tqdm jin du tiao
    for ai, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
        
        for pi, para in enumerate(article['paragraphs']):
            # wordss
            context = para['context']
            context = context.replace("''", '" ')
            # context = context.replace("``", '" ')
            contexts = nltk.sent_tokenize(context)

            xi = list(map(word_tokenize, contexts))
            # # split([-−—–/~"'“’”‘°])
            # xi = [process_tokens(tokens) for tokens in xi]  # process tokens
            # given xi, add chars

            # cxi = [[list(xijk) for xijk in xij] for xij in xi]
            
            for qa in para['qas']:

                question = word_tokenize(qa['question'])
                # question = process_tokens(question)
                q_id = qa["id"]
                
                if out_name != 'dev':

                    for answer in qa['answers']:
                        answer_text = answer['text']
                        answer_start = answer['answer_start']
                        answer_stop = answer_start + len(answer_text)

                        # TODO : put some function that gives word_start, word_stop here

                        # return yi0, yi1 = (num1, num2), (num3, num4)
                        # num1 : context中第num1个句子
                        # num2 : 第num1个句子的第num2个词
                        # num3 : context中第num3个句子
                        # num4 : 第num3个句子的第num4个词
                        yi0, yi1 = get_word_span(context, xi, answer_start, answer_stop)
                        # yi0 = answer['answer_word_start'] or [0, 0]
                        # yi1 = answer['answer_word_stop'] or [0, 1]
                        start = get_flat_idx(xi, yi0)
                        stop = get_flat_idx(xi, (yi1[0], yi1[1]-1))


                    fpw.write(' '.join(question)+'\n')
                    
                    fpw.write(str(start)+' '+str(stop)+' '+str(yi0[0])+' '+str(yi0[1])+' '+str(yi1[0])+' '+str(yi1[1]-1)+'\n')
                    for i in range(len(xi)):
                        fpw.write(' '.join(xi[i])+'\n')
                    fpw.write('\n')
                else:
                    fpw.write(' '.join(question)+'\n')
                    fpw.write(q_id+'\n')
                    for i in range(len(xi)):
                        fpw.write(' '.join(xi[i])+'\n')
                    fpw.write('\n')

        if args.debug:
            break


if __name__ == "__main__":
    main()