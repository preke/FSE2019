import torch
from torchtext import data
from torchtext import datasets
import time
import re
import os
from tqdm import tqdm

SOS_WORD = '<SOS>'
EOS_WORD = '<EOS>'
PAD_WORD = '<PAD>'


class MaxlenTranslationDataset(data.Dataset):
    # Code modified from
    # https://github.com/pytorch/text/blob/master/torchtext/datasets/translation.py
    # to be able to control the max length of the source and target sentences

    def __init__(self, path, fields, max_len=None, **kwargs):

        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]
        exts = ['src.txt', 'trg.txt']
        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with open(src_path) as src_file, open(trg_path) as trg_file:
            for src_line, trg_line in tqdm(zip(src_file, trg_file)):
                src_line, trg_line = src_line.split(' '), trg_line.split(' ')
                if max_len is not None:
                    src_line = src_line[:max_len]
                    src_line = str(' '.join(src_line))
                    trg_line = trg_line[:max_len]
                    trg_line = str(' '.join(trg_line))

                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))

        super(MaxlenTranslationDataset, self).__init__(examples, fields, **kwargs)


class DataPreprocessor(object):
    def __init__(self):
        self.text_field = self.generate_fields()
        # self.src_field, self.trg_field = self.generate_fields()

    def preprocess(self, train_path, val_path, train_file, val_file, max_len=None):
        # Generating torchtext dataset class
        print ("Preprocessing train dataset...")
        train_dataset = self.generate_data(train_path, max_len)

        print ("Saving train dataset...")
        self.save_data(train_file, train_dataset)

        print ("Preprocessing validation dataset...")
        val_dataset = self.generate_data(val_path, max_len)

        print ("Saving validation dataset...")
        self.save_data(val_file, val_dataset)

        # Building field vocabulary
        self.text_field.build_vocab(train_dataset, val_dataset)
        # self.src_field.build_vocab(train_dataset)
        # self.trg_field.build_vocab(train_dataset)

        # src_vocab, trg_vocab, src_inv_vocab, trg_inv_vocab = self.generate_vocabs()
        vocab, inv_vocab = self.generate_vocabs()
        # vocabs = {'src_vocab': src_vocab, 'trg_vocab': trg_vocab,
        #          'src_inv_vocab': src_inv_vocab, 'trg_inv_vocab': trg_inv_vocab}
        vocabs = {
            'vocab': vocab,
            'inv_vocab': inv_vocab
        }
        return train_dataset, val_dataset, vocabs

    def load_data(self, train_file, val_file):
        # Loading saved data
        train_dataset = torch.load(train_file)
        train_examples = train_dataset['examples']

        val_dataset = torch.load(val_file)
        val_examples = val_dataset['examples']

        # Generating torchtext dataset class
        # fields = [('src', self.src_field), ('trg', self.trg_field)]
        fields = [('src', self.text_field), ('trg', self.text_field)]
        train_dataset = data.Dataset(fields=fields, examples=train_examples)
        val_dataset = data.Dataset(fields=fields, examples=val_examples)

        # Building field vocabulary
        self.text_field.build_vocab(train_dataset, val_dataset)
        # self.src_field.build_vocab(train_dataset)
        # self.trg_field.build_vocab(train_dataset)

        # src_vocab, trg_vocab, src_inv_vocab, trg_inv_vocab = self.generate_vocabs()
        vocab, inv_vocab = self.generate_vocabs()
        # vocabs = {'src_vocab': src_vocab, 'trg_vocab': trg_vocab,
        #          'src_inv_vocab': src_inv_vocab, 'trg_inv_vocab': trg_inv_vocab}
        vocabs = {
            'vocab': vocab,
            'inv_vocab': inv_vocab
        }

        return train_dataset, val_dataset, vocabs

    def save_data(self, data_file, dataset):
        examples = vars(dataset)['examples']
        dataset = {'examples': examples}

        torch.save(dataset, data_file)

    def generate_fields(self):
        text_field = data.Field(
            init_token=SOS_WORD,
            eos_token=EOS_WORD,
            pad_token=PAD_WORD,
            include_lengths=True,
            batch_first=True)
        return text_field
        # src_field = data.Field(
        #     init_token=SOS_WORD,
        #     eos_token=EOS_WORD,
        #     pad_token=PAD_WORD,
        #     include_lengths=True,
        #     batch_first=True)
		#
        # trg_field = data.Field(
        #     init_token=SOS_WORD,
        #     eos_token=EOS_WORD,
        #     pad_token=PAD_WORD,
        #     include_lengths=True,
        #     batch_first=True)
        # return src_field, trg_field

    def generate_data(self, data_path, max_len=None):
        dataset = MaxlenTranslationDataset(
            path=data_path,
            fields=(self.text_field),
            max_len=max_len)

        return dataset

    def generate_vocabs(self):
        # Define string to index vocabs
        #src_vocab = self.src_field.vocab.stoi
        # trg_vocab = self.trg_field.vocab.stoi
        vocab = self.text_field.stoi

        # Define index to string vocabs
        #src_inv_vocab = self.src_field.vocab.itos
        # trg_inv_vocab = self.trg_field.vocab.itos

        inv_vocab = self.text_field.itos
        # return src_vocab, trg_vocab, src_inv_vocab, trg_inv_vocab
        return vocab, inv_vocab


