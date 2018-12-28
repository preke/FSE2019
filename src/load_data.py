# coding=utf-8
import os
import pandas as pd
import numpy as np
import re
import random
import urllib
from torchtext import data
from datetime import datetime
from sklearn.utils import shuffle
import torchtext.datasets as datasets
import pickle


def covert_to_tab(input_path, output_path):
    '''
    This function generate a tsv file of post & response
    for each instance in the source file
    '''
    file_writer = open(output_path, 'w')
    with open(input_path, 'r') as file_reader:
        for line in file_reader:
            try:
                tmp_list = line.split('***')
            except:
                print(line)
            file_writer.write(
                re.sub('[\s+\.\!\/_,$%^*(+\")]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+', ' ', tmp_list[4].strip()))
            file_writer.write('\t')
            file_writer.write(
                re.sub('[\s+\.\!\/_,$%^*(+\")]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+', ' ', tmp_list[5].strip()))
            file_writer.write('\n')
    file_writer.close()


def load_glove_as_dict(filepath):
    word_vec = {}
    with open(filepath) as fr:
        for line in fr:
            line = line.split()
            word = line[0]
            vec = line[1:]
            word_vec[word] = vec
    return word_vec


def gen_iter(path, text_field, args):
    '''
        Load TabularDataset from path,
        then convert it into a iterator
        return TabularDataset and iterator
    '''
    tmp_data = data.TabularDataset(
                            path=path,
                            format='tsv',
                            skip_header=False,
                            fields=[
                                    ('post', text_field),
                                    ('response', text_field)
                                    ]
                            )
    tmp_iter = data.BucketIterator(
                    tmp_data,
                    batch_size=args.batch_size,
                    sort_key=lambda x: len(x.question1),
                    device=args.device,
                    repeat=False)
    return tmp_data, tmp_iter


def load_data(args, TRAIN_TAB_PATH, VALID_TAB_PATH, TEST_TAB_PATH):
    text_field = data.Field(sequential=True, use_vocab=True, batch_first=True, lower=True)

    train_data, train_iter = gen_iter(TRAIN_TAB_PATH, text_field, args)
    valid_data, valid_iter = gen_iter(VALID_TAB_PATH, text_field, args)
    test_data, test_iter = gen_iter(TEST_TAB_PATH, text_field, args)

    return text_field, \
           train_data, train_iter, \
           valid_data, valid_iter, \
           test_data, test_iter

