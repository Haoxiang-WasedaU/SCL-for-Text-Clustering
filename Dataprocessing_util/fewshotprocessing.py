# -*- coding: utf-8 -*-
# -*- coding: UTF-8 -*-
# !/usr/bin/python
import os, csv, random, torch, torch.nn as nn, numpy as np
import torch
from torchtext import data
# few-shot data processing
TEXT1 = data.Field(lower=True, tokenize='spacy', tokenizer_language='en_core_web_sm', batch_first=True)
path ='/home/hollis_shi/Clustering'
LABEL = data.LabelField(dtype=torch.long)
train_data = data.TabularDataset.splits(
    path=path, train='20groupnewsback.tsv', format='tsv',
    fields=[('text1', TEXT1), ('text2', TEXT1),('text3',TEXT1),('label', LABEL)])[0]
TEXT1.build_vocab(train_data,
                  max_size=20000,
                  vectors='glove.6B.300d',
                   unk_init=torch.Tensor.normal_
)
INPUT_DIM = 400002
EMBEDDING_DIM = 300
Hidden_dim=100
PAD_IDX = TEXT1.vocab.stoi[TEXT1.pad_token]
UNK_IDX = TEXT1.vocab.stoi[TEXT1.unk_token]
train_iter = data.BucketIterator(train_data, batch_size=50000, sort_key=lambda x:len(x.text1), sort_within_batch=True)# 训练集需要shuffle，但因为验证测试集不需要 # 可以生成验证和测试集的迭代器直接用data.iterator.Iterator类就足够了 )
pred_trained = TEXT1.vocab.vectors
Q = np.array(pred_trained.detach().numpy())
Q[0]= torch.zeros(300)
Q[1]= torch.zeros(300)
print(Q.shape)
LABEL.build_vocab(train_data)
np.savetxt('pred_trained_stackoverflow_cl', Q)
Stack_text0 = []
Stack_text1 = []
Stack_text2 = []
Stack_text3 = []
Stack_text4 = []
Stack_text5 = []
Stack_text6 = []
Stack_text7 = []
Stack_text8 = []
Stack_text9 = []
Stack_text10 = []
Stack_text11 = []
Stack_text12 = []
Stack_text13 = []
Stack_text14 = []
Stack_text15 = []
Stack_text16 = []
Stack_text17 = []
Stack_text18 = []
Stack_text19 = []
Maxlen=0
for batch in train_iter:
    a = batch.text2.cpu().numpy().tolist()
    Maxlen = len(a)
    c = batch.text1.cpu().numpy().tolist()
    label = batch.label.cpu().numpy().tolist()
    with open('data/StackOverflow_clunsupervised_lstm', 'a') as f2:
        for i in range(len(c)):
            if int(len(c)/10*3)<i<int(len(c)/10*5):
                with open('data/StackOverflow_cltest_lstm', 'a') as p:
                    p.write(str(a[i][:128] + [PAD_IDX] * (128 - len(c[i]) - 1))+"\t"+str(label[i]))
                    p.write("\n")
            else:
                if label[i] == 0:
                    Stack_text0.append(a[i][:128] + [PAD_IDX] * (128 - len(a[i])))
                if label[i] == 1:
                    Stack_text1.append(a[i][:128] + [PAD_IDX] * (128 - len(a[i])))
                if label[i] == 2:
                    Stack_text2.append(a[i][:128] + [PAD_IDX] * (128 - len(a[i])))
                if label[i] == 3:
                    Stack_text3.append(a[i][:128] + [PAD_IDX] * (128 - len(a[i])))
                if label[i] == 4:
                    Stack_text4.append(a[i][:128] + [PAD_IDX] * (128 - len(a[i])))
                if label[i] == 5:
                    Stack_text5.append(a[i][:128] + [PAD_IDX] * (128 - len(a[i])))
                if label[i] == 6:
                    Stack_text6.append(a[i][:128] + [PAD_IDX] * (128 - len(a[i])))
                if label[i] == 7:
                    Stack_text7.append(a[i][:128] + [PAD_IDX] * (128 - len(a[i])))
                if label[i] == 8:
                    Stack_text8.append(a[i][:128] + [PAD_IDX] * (128 - len(a[i])))
                if label[i] == 9:
                    Stack_text9.append(a[i][:128] + [PAD_IDX] * (128 - len(a[i]) ))
                if label[i] == 10:
                    Stack_text10.append(a[i][:128] + [PAD_IDX] * (128 - len(a[i]) ))
                if label[i] == 11:
                    Stack_text11.append(a[i][:128] + [PAD_IDX] * (128 - len(a[i]) ))
                if label[i] == 12:
                    Stack_text12.append(a[i][:128] + [PAD_IDX] * (128 - len(a[i]) ))
                if label[i] == 13:
                    Stack_text13.append(a[i][:128] + [PAD_IDX] * (128 - len(a[i]) ))
                if label[i] == 14:
                    Stack_text14.append(a[i][:128] + [PAD_IDX] * (128 - len(a[i]) ))
                if label[i] == 15:
                    Stack_text15.append(a[i][:128] + [PAD_IDX] * (128 - len(a[i]) ))
                if label[i] == 16:
                    Stack_text16.append(a[i][:128] + [PAD_IDX] * (128 - len(a[i]) ))
                if label[i] == 17:
                    Stack_text17.append(a[i][:128] + [PAD_IDX] * (128 - len(a[i]) ))
                if label[i] == 18:
                    Stack_text18.append(a[i][:128] + [PAD_IDX] * (128 - len(a[i]) ))
                if label[i] == 19:
                    Stack_text19.append(a[i][:128] + [PAD_IDX] * (128 - len(a[i]) ))
                q1= c[i][:128] + [PAD_IDX] * (128 - len(c[i]))
                q2= a[i][:128] + [PAD_IDX] * (128 - len(a[i]))
                f2.write(str(q1) + '\t' + str(q2) + '\n')

text1 = []
text2 = []
for num in range(int(Maxlen/10/20)):
    text1.append(Stack_text0[num])
    text2.append(Stack_text0[num + 1])
    text1.append(Stack_text1[num])
    text2.append(Stack_text1[num + 1])
    text1.append(Stack_text2[num])
    text2.append(Stack_text2[num + 1])
    text1.append(Stack_text3[num])
    text2.append(Stack_text3[num + 1])
    text1.append(Stack_text4[num])
    text2.append(Stack_text4[num + 1])
    text1.append(Stack_text5[num])
    text2.append(Stack_text5[num + 1])
    text1.append(Stack_text6[num])
    text2.append(Stack_text6[num + 1])
    text1.append(Stack_text7[num])
    text2.append(Stack_text7[num + 1])
    text1.append(Stack_text8[num])
    text2.append(Stack_text8[num + 1])
    text1.append(Stack_text9[num])
    text2.append(Stack_text9[num + 1])
    text1.append(Stack_text10[num])
    text2.append(Stack_text10[num + 1])
    text1.append(Stack_text11[num])
    text2.append(Stack_text11[num + 1])
    text1.append(Stack_text12[num])
    text2.append(Stack_text12[num + 1])
    text1.append(Stack_text13[num])
    text2.append(Stack_text13[num + 1])
    text1.append(Stack_text14[num])
    text2.append(Stack_text14[num + 1])
    text1.append(Stack_text15[num])
    text2.append(Stack_text15[num + 1])
    text1.append(Stack_text16[num])
    text2.append(Stack_text16[num + 1])
    text1.append(Stack_text17[num])
    text2.append(Stack_text17[num + 1])
    text1.append(Stack_text18[num])
    text2.append(Stack_text18[num + 1])
    text1.append(Stack_text19[num])
    text2.append(Stack_text19[num + 1])
with open('data/StackOverflow_self_learning_LSTM.txt', 'w') as f:
    for i in range(len(text1)):
        print(text1[i],text2[i])
        f.write(str(text1[i])+"\t"+str(text2[i]))
        f.write('\n')
        if i % 100 == 0:
            print("这是第", i, "条数据")

