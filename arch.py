import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_emb, dim_out=2):
        super(SelfAttention, self).__init__()
        self.dim_emb = dim_in
        self.query = nn.Linear(dim_in, dim_emb)
        self.key = nn.Linear(dim_in, dim_emb)
        self.value = nn.Linear(dim_in, dim_emb)
        self.softmax = nn.Softmax(dim=dim_out)

        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.dim_emb ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted
        

class MLP(torch.nn.Module):

    def __init__(self, dim_in, dim_h, dim_out=2):
        super(MLP, self).__init__()
        self.fc = nn.Linear(dim_in, dim_h)
        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(dim_h)
        self.dense = nn.Linear(dim_h, dim_out)

    def forward(self, x):
        x = x.to(torch.float)
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.dense(x)
        return x

class SelfAttentionClassifier(torch.nn.Module):

    def __init__(self, dim_in, dim_seq, dim_h0, dim_emb, dim_h1, dim_out=2, dropout=0.1):
        super(SelfAttentionClassifier, self).__init__()

        self.dim_in = dim_in
        self.dim_emb = dim_emb
        self.dim_seq = dim_seq
        self.linear_in = nn.Linear(dim_in, dim_h0)
        self.pos_enc = PositionalEncoding(dim_h0, n_position=dim_seq)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.attention = SelfAttention(dim_h0, dim_emb, dim_out)
        self.fc = nn.Linear(dim_seq*dim_emb, dim_h1)
        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(dim_h1)
        self.linear_out = nn.Linear(dim_h1, dim_out)

    def forward(self, x):
        dim_b, dim_s, dim_v = x.shape
        assert dim_v == self.dim_in
        assert dim_s == self.dim_seq
        x = x.to(torch.float)
        x = self.linear_in(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        x = self.attention(x)
        x = x.view(dim_b, self.dim_seq*self.dim_emb)
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.linear_out(x)

        return x

