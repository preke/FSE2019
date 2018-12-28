import torch
import torchtext
import argparse
import pandas as pd
import numpy as np
from load_data import covert_to_tab, load_data


# PATH
TRAIN_PATH = '../data/single_train_data.txt'
VALID_PATH = '../data/single_valid_data.txt'
TEST_PATH = '../data/single_test_data.txt'

TRAIN_TAB_PATH = '../data/single_train_data.tsv'
VALID_TAB_PATH = '../data/single_valid_data.tsv'
TEST_TAB_PATH = '../data/single_test_data.tsv'

# preprocess
covert_to_tab(input_path=TRAIN_PATH, output_path=TRAIN_TAB_PATH)
covert_to_tab(input_path=VALID_PATH, output_path=VALID_TAB_PATH)
covert_to_tab(input_path=TEST_PATH, output_path=TEST_TAB_PATH)

# Parameters
parser = argparse.ArgumentParser(description='')
parser.add_argument('-lr', type=float, default=0.005, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
args = parser.parse_args()
args.device = torch.device('cuda')
# load data
text_field, train_data, train_iter, valid_data, valid_iter, test_data, test_iter = \
    load_data(args, TRAIN_TAB_PATH, VALID_TAB_PATH, TEST_TAB_PATH)

text_field.build_vocab(train_data, valid_data)
# train model