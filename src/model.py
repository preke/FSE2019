import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class lstm(nn.Module):
    def __init__(self, args):
        super(lstm, self).__init__()
        self.args = args

        # for word embeddings
        self.V = args.word_embedding_num
        self.D = args.word_embedding_length
        self.word_embedding = nn.Embedding(self.V, self.D)
        self.bidirectional = False
        # use pre-trained
        if args.word_Embedding:
            pretrained_weight = np.array(args.pretrained_weight)
            self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        self.lstm = nn.LSTM(input_size=self.D,
                            hidden_size=self.hidden_dim,
                            batch_first=True,
                            num_layers=self.num_layers)
        # In train function, set args.batch_size as ba.size()[0]
        self.hidden = self.init_hidden(self.num_layers, args.batch_size)

    def init_hidden(self, batch_size, device):
        '''
        Initialize hidden layers with random nums
        '''
        layer_num = 2 if self.bidirectional else 1
        if device == -1:
            return (Variable(torch.randn(layer_num, batch_size, self.hidden_dim // layer_num)), \
                    Variable(torch.randn(layer_num, batch_size, self.hidden_dim // layer_num)))
        else:
            return (Variable(torch.randn(layer_num, batch_size, self.hidden_dim // layer_num).cuda()), \
                    Variable(torch.randn(layer_num, batch_size, self.hidden_dim // layer_num).cuda()))

    def forward(self, sentence, hidden):
        sentence = self.word_embedding(sentence)
        # Outputs: output, (h_n, c_n)

        lstm_out, (lstm_h, lstm_c) = self.lstm(sentence, hidden)
        return lstm_h


class lstm_similarity(lstm):
    def __init__(self, args):
        super(lstm_similarity, self).__init__()

    def forward(self, s1, s2, hidden=None):
        '''
            Embed each sentence with the last hidden layer output;
            calculate (cosine)similarity between these 2 sentences
        '''
        s1 = self.word_embedding(s1)
        s1_out, (s1_h, s1_c) = self.lstm(s1, hidden)
        s1_embedded = s1_h

        s2 = self.word_embedding(s2)
        s2_out, (s2_h, s2_c) = self.lstm(s2, hidden)
        s2_embedded = s2_h

        return F.cosine_similarity(s1_embedded, s2_embedded)


class Bi_lstm(lstm):
    def __init__(self, args):
        super(Bi_lstm, self).__init__()
        self.bidirectional = True
        self.lstm = nn.LSTM(input_size=self.D,
                            hidden_size=self.hidden_dim,
                            batch_first=True,
                            num_layers=self.num_layers,
                            bidirectional=True)

    def forward(self, sentence):
        sentence = self.word_embedding(sentence)
        lstm_out, (lstm_h, lstm_c) = self.lstm(sentence)
        sentence = torch.cat((lstm_h[0], lstm_h[1]), dim=1)
        return sentence


class Bi_lstm_similarity(lstm):
    def __init__(self, args):
        super(Bi_lstm_similarity, self).__init__()
        self.bidirectional = True
        self.lstm = nn.LSTM(input_size=self.D,
                            hidden_size=self.hidden_dim,
                            batch_first=True,
                            num_layers=self.num_layers,
                            bidirectional=True)

    def forward(self, s1, s2):
        s1 = self.word_embedding(s1)
        s1_out, (s1_h, s1_c) = self.lstm(s1)
        s1 = torch.cat((s1_h[0], s1_h[1]), dim=1)

        s2 = self.word_embedding(s2)
        s2_out, (s2_h, s2_c) = self.lstm(s2)
        s2 = torch.cat((s2_h[0], s2_h[1]), dim=1)

        return F.cosine_similarity(s1, s2)
