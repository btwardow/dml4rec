import torch
from torch import nn
from torch.nn import functional as F

ACTIVATIONS = {'iden': lambda x: x, 'relu': torch.relu, 'tanh': torch.tanh, 'sigm': torch.sigmoid}


class SimpleMean(torch.nn.Module):
    def __init__(self, rec):
        super().__init__()
        self.embed = nn.Embedding(rec.items_num, rec.emb_dim, max_norm=rec.max_emb_norm, padding_idx=0)
        torch.nn.init.uniform_(self.embed.weight)
        with torch.no_grad():
            self.embed.weight[0].fill_(0)

    def forward(self, x):
        s_z = self.embed(x)
        return s_z.mean(1)


class MaxPool(torch.nn.Module):
    def __init__(self, rec, activ='tanh', dropout1_rate=0.0, dropout2_rate=0.0, agg='max'):
        super().__init__()
        self.dropout1_rate = dropout1_rate
        self.dropout2_rate = dropout2_rate
        self.embed = nn.Embedding(rec.items_num, rec.emb_dim, max_norm=rec.max_emb_norm, padding_idx=0)
        if agg == 'max':
            self.gavg = lambda x: x.max(1)[0]
        elif agg == 'avg':
            self.gavg = lambda x: x.mean(1)
        else:
            raise RuntimeError(f'Bad aggregation: {agg}!')
        self.dropout1 = nn.Dropout(dropout1_rate)
        self.sfc1 = torch.nn.Linear(rec.emb_dim, rec.emb_dim)
        self.dropout2 = nn.Dropout(dropout2_rate)
        self.sfc2 = torch.nn.Linear(rec.emb_dim, rec.emb_dim)
        self.s_activ = ACTIVATIONS[activ]

    def forward(self, x):
        s_z = self.embed(x)
        s_z = self.gavg(s_z)
        s_z = self.dropout1(s_z)
        s_z = self.sfc1(s_z)
        s_z = self.s_activ(s_z)
        s_z = self.dropout2(s_z)
        s_z = self.sfc2(s_z)
        s_z = self.s_activ(s_z)
        return s_z


class WeightedPool(torch.nn.Module):
    def __init__(self, rec, activ='tanh'):
        super().__init__()
        self.embed = nn.Embedding(rec.items_num, rec.emb_dim, max_norm=rec.max_emb_norm, padding_idx=0)
        self.weight = torch.linspace(0.1, 1.0, rec.max_len)
        self.sfc1 = torch.nn.Linear(rec.emb_dim, rec.emb_dim)
        self.sfc2 = torch.nn.Linear(rec.emb_dim, rec.emb_dim)
        self.s_activ = ACTIVATIONS[activ]

    def forward(self, x):
        s_z = self.embed(x)
        # w = F.softmax(self.weight)
        s_z = torch.matmul(self.weight, s_z)
        s_z = F.normalize(s_z, p=2, dim=-1)
        s_z = self.sfc1(s_z)
        s_z = self.s_activ(s_z)
        s_z = self.sfc2(s_z)
        s_z = self.s_activ(s_z)
        return s_z

    def to(self, device):
        self.weight = self.weight.to(device)
        return super().to(device)


class OnlyLastItem(nn.Module):
    def __init__(self, rec, activ='tanh'):
        super().__init__()
        self.embed = nn.Embedding(rec.items_num, rec.emb_dim, max_norm=rec.max_emb_norm)
        self.sfc1 = torch.nn.Linear(rec.emb_dim, rec.emb_dim)
        self.s_activ = ACTIVATIONS[activ]

    def forward(self, x):
        s_z = self.embed(x[:, -1])
        s_z = self.sfc1(s_z)
        s_z = self.s_activ(s_z)
        return s_z


class TagSpace(nn.Module):
    """
    https://github.com/xiaoyizy/TagSpace-pytorch
    """
    def __init__(self, rec, filter_sizes='3', num_filters=256):
        super().__init__()
        self.vocab_size = rec.items_num  # N in paper
        self.embedding_size = rec.emb_dim  # d in paper
        self.embed = nn.Embedding(
            self.vocab_size, self.embedding_size, max_norm=rec.max_emb_norm, padding_idx=0
        )  # or use w2v/wsabie as pretrained lookup
        self.max_seq_length = rec.max_len  # l in paper
        filters = list(map(int, filter_sizes.split(',')))
        assert len(filters) == 1
        self.window_size = filters[0]  # k in paper
        assert self.window_size < self.max_seq_length
        self.hidden_size = num_filters  # H in paper

        self.conv = nn.Conv1d(
            in_channels=self.embedding_size, out_channels=self.hidden_size, kernel_size=self.window_size, padding=1
        )
        self.maxpool = nn.MaxPool1d(kernel_size=self.max_seq_length)
        self.decoder = nn.Linear(in_features=self.hidden_size, out_features=self.embedding_size, bias=True)

    def forward(self, input):
        batch_size = input.size(0)
        # 1. get embed
        post_embed = self.embed(input)
        #		 print('Size after emebedding:', post_embed.size())
        # 2. convolution + tanh activation
        post_conv = torch.tanh(self.conv(post_embed.permute(0, 2, 1)))
        #		 print('Size after convolution layer:', post_conv.size())
        # 3. maxpool + tanh activation
        post_maxpool = torch.tanh(self.maxpool(post_conv).reshape(batch_size, self.hidden_size))
        #		 print('Size after max pooling:', post_maxpool.size())
        # 4. linear decoder
        tweets_embed = self.decoder(post_maxpool)
        #		 print('Size of output:', tweets_embed.size())
        return tweets_embed


class CNN1D(nn.Module):
    def __init__(
        self,
        rec,
        activ='tanh',
        dropout_rate=0.5,
        num_filters=256,
        filter_sizes='1,3',
    ):
        super().__init__()
        self.activ_fun = ACTIVATIONS[activ]
        self.max_len = rec.max_len
        self.embed = nn.Embedding(rec.items_num, rec.emb_dim, max_norm=rec.max_emb_norm, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout_rate)
        self.conv_pool_layers = []
        self.conv_pool_output_size = 0
        for filter_size in list(map(int, filter_sizes.split(','))):
            self.conv_pool_layers.append(
                (
                    nn.Conv1d(in_channels=rec.emb_dim, out_channels=num_filters, kernel_size=filter_size,
                              padding=1).to(rec._device), nn.AdaptiveAvgPool1d(rec.max_len).to(rec._device)
                )
            )
            self.conv_pool_output_size += rec.max_len * num_filters

        # self.pool_flatten = Flatten(data_format='channels_last', name='flatten')
        self.pool_dropout = nn.Dropout(dropout_rate)
        self.pool_dense1 = nn.Linear(self.conv_pool_output_size, rec.emb_dim)
        # self.pool_dense2 = nn.Linear(rec.emb_dim, rec.emb_dim)

    def forward(self, x):
        batch_size, max_len = x.size()
        z = self.embed(x)
        z = self.emb_dropout(z)
        pool_outputs = []
        for c, p in self.conv_pool_layers:
            _z = c(z.permute(0, 2, 1))
            _z = p(_z)
            _z = self.activ_fun(_z)
            pool_outputs.append(_z)
        z = torch.cat(pool_outputs, dim=-1)
        z = z.reshape(batch_size, -1)
        z = self.pool_dropout(z)
        z = self.pool_dense1(z)
        z = self.activ_fun(z)
        # z = self.pool_dense2(z)
        return z


class TextCNN(nn.Module):
    """
    Model from:
    https://github.com/Shawn1993/cnn-text-classification-pytorch

    Article:
    Convolutional Neural Networks for Sentence Classification
    https://arxiv.org/abs/1408.5882
    """
    def __init__(
        self,
        rec,
        filter_sizes='1,3,5',
        num_filters=256,
        dropout_rate=0.5,
    ):
        super().__init__()
        self.num_filters = num_filters
        V = rec.items_num
        D = rec.emb_dim
        C = rec.emb_dim
        Ci = 1
        Co = self.num_filters
        Ks = list(map(int, filter_sizes.split(',')))

        self.embed = nn.Embedding(V, D, max_norm=rec.max_emb_norm, padding_idx=0)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit


class RNN(nn.Module):
    def __init__(self, rec, hidden_size=None, activ='tanh', rnn_type='GRU', rnn_dropout_rate=0.0, dropout1_rate=0.2):
        super().__init__()
        if hidden_size is None:
            hidden_size = rec.emb_dim
        self.embed = nn.Embedding(rec.items_num, rec.emb_dim, max_norm=rec.max_emb_norm, padding_idx=0)
        self.rnn_dropout_rate = rnn_dropout_rate
        self.rnn_type = rnn_type
        RNNUnit = eval(f'torch.nn.{rnn_type}')
        self.rnn = RNNUnit(rec.emb_dim, hidden_size, batch_first=True, dropout=rnn_dropout_rate)
        self.dropout1 = nn.Dropout(dropout1_rate)
        self.sfc1 = torch.nn.Linear(hidden_size, rec.emb_dim)
        self.s_activ = ACTIVATIONS[activ]

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.dropout1(x)
        x = self.sfc1(x)
        x = self.s_activ(x)
        return x
