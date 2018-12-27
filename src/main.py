import torch
import torchtext
from load_data import covert_to_tab


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

# load data

# train model