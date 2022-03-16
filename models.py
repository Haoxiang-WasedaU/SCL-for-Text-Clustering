# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


""" Transformer Model Classes & Config Class """

import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import split_last, merge_last

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Config(NamedTuple):
    "Configuration for BERT model"
    vocab_size: int = None # Size of Vocabulary
    dim: int = 768 # Dimension of Hidden Layer in Transformer Encoder
    n_layers: int = 12 # Numher of Hidden Layers
    n_heads: int = 12 # Numher of Heads in Multi-Headed Attention Layers
    dim_ff: int = 768*4 # Dimension of Intermediate Layers in Positionwise Feedforward Net
    #activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    p_drop_hidden: float = 0.1 # Probability of Dropout of various Hidden Layers
    p_drop_attn: float = 0.1 # Probability of Dropout of Attention Layers
    max_len: int = 512 # Maximum Length for Positional Embeddings
    n_segments: int = 2 # Number of Sentence Segments

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.dim))
        self.beta  = nn.Parameter(torch.zeros(cfg.dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."
    def __init__(self, cfg):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim) # token embedding
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.dim) # position embedding
        self.seg_embed = nn.Embedding(cfg.n_segments, cfg.dim) # segment(token type) embedding

        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x) # (S,) -> (1, S) -> (B, S)  이렇게 외부에서 생성되는 값

        e = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.drop(self.norm(e))


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.dim, cfg.dim)
        self.proj_k = nn.Linear(cfg.dim, cfg.dim)
        self.proj_v = nn.Linear(cfg.dim, cfg.dim)
        self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, cfg.dim_ff)
        self.fc2 = nn.Linear(cfg.dim_ff, cfg.dim)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class Block(nn.Module):
    """ Transformer Block """
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, mask):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])   # h 번 반복

    def forward(self, x, seg, mask):
        h = self.embed(x, seg)
        for block in self.blocks:
            h = block(h, mask)
        return h


class CNNClassifier(nn.Module):
    def __init__(self, cfg, n_labels):
        super(CNNClassifier, self).__init__()
        self.embed = nn.Embedding(25000+2, 300)
        # weight = LoadPreTrain(cfg.glove, cfg.vocab)
        # self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(weight), freeze=False).to(device)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=300,
                out_channels=cfg.n_filters,
                kernel_size=ks
            )
            for ks in cfg.kernel_size
        ])
        self.dropout = nn.Dropout(cfg.p_drop_hidden)
        self.fc = nn.Linear(len(cfg.kernel_size) * cfg.n_filters, n_labels)

    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.embed(text)

        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.permute(0, 2, 1)

        # embedded = [batch size, emb dim, sent len]
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

class Lstm(nn.Module):
    def __init__(self, x_dim, e_dim, h_dim,o_dim,drop_out,pad_idx):
        super(Lstm, self).__init__()
        self.num = 0
        self.embedding_metrix= torch.FloatTensor(np.loadtxt('pred_trained'))
        self.word_embeddings = nn.Embedding(x_dim, e_dim, padding_idx=pad_idx).from_pretrained(self.embedding_metrix)
        embedding_matrix = torch.LongTensor().cuda()
        self.lstm = nn.LSTM(e_dim, e_dim, bidirectional=False, batch_first=False)
        self.decoder = nn.Linear(e_dim, o_dim)
        self.dropout = nn.Dropout(drop_out)
        self.activ = nn.Tanh()

    def forward(self, x):

        embeds = self.word_embeddings(x)
        embeds= self.dropout(embeds)
        embeds = embeds.transpose(0,1)
        out, (h_n,c_n) = self.lstm(embeds)
        y = self.activ(self.decoder(out))
        y = self.dropout(y)

        y = y.sum(0)
        return y

class Classifier(nn.Module):
    """ Classifier with Transformer """
    def __init__(self, cfg, n_labels=20):
        super().__init__()
        self.transformer = Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, n_labels)
        self.softmax= nn.Softmax(dim=1)
        self.init_param()
        self.h_dim = 20
        self.classifier = nn.Linear(cfg.dim, 20)
        self.embedding = 768
        self.conv1 = nn.Conv2d(1, self.h_dim, (3, self.embedding))
        self.conv2 = nn.Conv2d(1, self.h_dim, (4, self.embedding))
        self.conv3 = nn.Conv2d(1, self.h_dim, (5, self.embedding))
        self.model = nn.Sequential(
            nn.Linear(60, 20),
            nn.Tanh(),
            nn.ReLU(),
            nn.Linear(20, 20)
        )

    def init_param(self):
        self.classifier.weight.data.normal_(std=0.02)
        self.classifier.bias.data.fill_(0)

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activ(self.fc(h[:, 0]))  # 맨앞의 [CLS]만 뽑아내기
        logits = self.classifier(self.drop(pooled_h))
        return logits


class Classforunsup(nn.Module):
    """ Classifier with Transformer """
    def __init__(self, cfg, n_labels):
        super().__init__()
        self.transformer = Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, n_labels)
        self.softmax= nn.Softmax(dim=1)
        self.init_param()

    def init_param(self):
        self.classifier.weight.data.normal_(std=0.02)
        self.classifier.bias.data.fill_(0)

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activ(self.fc(h[:, 0]))
        logits = self.classifier(self.drop(pooled_h))
        return logits

class Opinion_extract(nn.Module):
    """ Opinion_extraction """
    def __init__(self, cfg, max_len, n_labels):
        super().__init__()
        self.transformer = Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.extract = nn.Linear(cfg.dim, n_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.transformer(input_ids, segment_ids, input_mask)
        h = self.drop(self.activ(self.fc(h[:, 1:-1])))
        seq_h = self.extract(h)
        seq_h = seq_h.squeeze()
        return self.sigmoid(seq_h)


def Word2Id(vocab_file):
    f = open(vocab_file, 'r', encoding='utf-8')
    word2id = dict()
    id2word = dict()
    for ix, row in enumerate(f.readlines()):
        line = row.strip('\n')
        word2id[line] = ix
        id2word[ix] = line
    return word2id, id2word


def GetPreTrain(glove, vocab_file, tmp_file):
    word2id, id2word = Word2Id(vocab_file=vocab_file)
    glove_file = datapath(glove)
    tmp_file = get_tmpfile('./word2vec.txt')

    glove2word2vec(glove_file, tmp_file)

    wvmodel = KeyedVectors.load_word2vec_format(tmp_file)
    vocab_size = len(word2id.keys())+1

    embed_size = 300
    weight = torch.zeros(vocab_size+1, embed_size)
    for i in range(len(wvmodel.index2word)):
        try:
            index = word2id[wvmodel.index2word[i]]
        except:
            continue

        weight[index, :] = torch.from_numpy(wvmodel.get_vector(
            id2word[word2id[wvmodel.index2word[i]]]
        ))
    print(weight)
    exit()
    return weight


def LoadPreTrain(glove, vocab_file):
    word2vec = {}
    pretrain_embedding = []
    with open(glove, 'r') as input_data:
        for i, line in enumerate(input_data.readlines()):
            if i != 0:
                word2vec[line.split()[0]] = [float(j) for j in line.split()[1:]]
    unk = []
    unk.extend([0.0]*300)
    pretrain_embedding.append(unk)
    word2id, id2word = Word2Id(vocab_file=vocab_file)

    for word in word2id:
        if word in word2vec:
            pretrain_embedding.append(word2vec[word])
        else:
            pretrain_embedding.append(unk)

    pretrain_embedding = np.asarray(pretrain_embedding)
    return pretrain_embedding
