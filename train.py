# Copyright 2019 SanghunYun, Korea University.
# (Strongly inspired by Dong-Hyun Lee, Kakao Brain)
#
# Except load and save function, the whole codes of file has been modified and added by
# SanghunYun, Korea University for UDA.
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
import utils.Bcubed_news as Bucued
from sklearn import metrics
import numpy as np
import os
# import bcubed
from sklearn.metrics import silhouette_score
import matplotlib.pylab as pylab
import json
from copy import deepcopy
from typing import NamedTuple
from sklearn.cluster import k_means, KMeans
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

from utils import checkpoint
# from utils.logger import Logger
from tensorboardX import SummaryWriter
from utils.utils import output_logging

def cluster_acc(Y_pred, Y):
    from scipy.optimize import linear_sum_assignment as linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1)
    w = np.zeros((int(D), int(D)))
    for i in range(Y_pred.size):
        # print(Y_pred[i],int(Y[i]))
        w[Y_pred[i], int(Y[i])] += 1
        ind = linear_assignment(w.max() - w)
        indices = np.asarray(ind)
        ind = np.transpose(indices)
    return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, w

class Trainer(object):
    """Training Helper class"""

    def __init__(self, cfg, model, data_iter, optimizer, device):
        self.cfg = cfg
        self.model = model
        print(type(model))
        print(self.model)
        self.optimizer = optimizer
        self.device = device
        if len(data_iter) == 1:
            self.sup_iter = data_iter[0]
        elif len(data_iter) == 2:
            self.sup_iter = self.repeat_dataloader(data_iter[0])
            self.unsup_iter = self.repeat_dataloader(data_iter[1])
        elif len(data_iter) == 3:
            self.sup_iter = self.repeat_dataloader(data_iter[0])
            self.unsup_iter = self.repeat_dataloader(data_iter[1])
            self.eval_iter = data_iter[2]

    def train(self, get_loss, get_acc, model_file, pretrain_file):
        """ train uda"""

        if self.cfg.results_dir:
            logger = SummaryWriter(log_dir=os.path.join(self.cfg.results_dir, 'logs'))
        model = self.model.to(self.device)
        if self.cfg.data_parallel:  # Parallel GPU mode
            model = nn.DataParallel(model)

        global_step = 0
        loss_sum = 0.
        clustering_Max, AmiMax, NmiMax, AriMax=0,0,0,0
        Max_Acc_global_step, Spectral_Acc, Kmeans_Acc, Spectral_AMI, Kmeans_AMI, Spectral_ARI, Kmeans_ARI,Spectral_Nmi, Kmeans_Nmi= 0,0,0,0,0,0,0,0,0

        Max_sc_score_class5,Max_sc_score_class10,Max_sc_score_class15,Max_sc_score_class20,Max_sc_score_class25,Max_sc_score_class30=0,0,0,0,0,0
        Max_calinski_harabasz_score_class5,Max_calinski_harabasz_score_class10,Max_calinski_harabasz_score_class15,Max_calinski_harabasz_score_class20,Max_calinski_harabasz_score_class25,Max_calinski_harabasz_score_class30=0,0,0,0,0,0
        Max_cube5,Max_cube10,Max_cube15,Max_cube20=0,0,0,0
        iter_bar = tqdm(self.unsup_iter, total=self.cfg.total_steps) if self.cfg.uda_mode \
            else tqdm(self.sup_iter, total=self.cfg.total_steps)
        for i, batch in enumerate(iter_bar):

            # Device assignment
            if self.cfg.uda_mode:
                sup_batch = [t.to(self.device) for t in next(self.sup_iter)]
                unsup_batch = [t.to(self.device) for t in batch]
            else:
                sup_batch = [t.to(self.device) for t in batch]
                unsup_batch = None
            # update
            self.optimizer.zero_grad()
            final_loss, sup_loss, unsup_loss= get_loss(model, sup_batch, unsup_batch, global_step)
            # sup_loss.backward()
            final_loss.item().backward()
            self.optimizer.step()
            global_step += 1
            loss_sum += final_loss.item()
            if self.cfg.uda_mode:
                if unsup_loss is None:
                    iter_bar.set_description('final=%5.3f ' \
                                         % (final_loss.item()))
                else:
                    iter_bar.set_description('final=%5.3f unsup=%5.3f sup=%5.3f' \
                                             % (final_loss.item(), unsup_loss.item(), sup_loss.item()))

            else:
                iter_bar.set_description('loss=%5.3f' % (final_loss.item()))

            # logging
            if self.cfg.uda_mode:
                if sup_loss or unsup_loss is None:
                    logger.add_scalars('data/scalar_group',
                                       {'final_loss': final_loss.item(),
                                        'lr': self.optimizer.get_lr()[0]
                                        }, global_step)
                else:
                    logger.add_scalars('data/scalar_group',
                                   {'final_loss': final_loss.item(),
                                    'sup_loss': sup_loss.item(),
                                    'unsup_loss': unsup_loss.item(),
                                    'lr': self.optimizer.get_lr()[0]
                                    }, global_step)
            else:
                logger.add_scalars('data/scalar_group',
                                   {'sup_loss': final_loss.item()}, global_step)

            if global_step % self.cfg.save_steps == 0:
                self.save(global_step)
            if get_acc and global_step % self.cfg.check_steps == 0 and global_step > 1500:
                text1, label1, results = self.clustereval(get_acc, None, model)
                dictclass5 = {'0': [], '1': [], '2': [], '3': [], '4': []}
                dictclass10 = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}
                dictclass15 = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [],
                           '10': [], '11': [], '12': [], '13': [], '14': []}
                dictclass20 = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [],
                           '10': [], '11': [], '12': [], '13': [], '14': [], '15': [], '16': [], '17': [], '18': [],
                           '19': []}
                dictlabel = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [],
                         '10': [], '11': [], '12': [], '13': [], '14': [], '15': [], '16': [], '17': [], '18': [],
                         '19': []}
                total_accuracy = torch.cat(results).mean().item()
                text1 = text1[128:]
                label = label1[128:]
                max_acc = [0., 0, 0.,0.,0.,0.]
                Class5= KMeans(n_clusters=5).fit_predict(text1)
                Class10 = KMeans(n_clusters=10).fit_predict(text1)
                Class15 = KMeans(n_clusters=15).fit_predict(text1)
                Class20 = KMeans(n_clusters=20).fit_predict(text1)
                y2_pred = KMeans(n_clusters=20,random_state=9).fit_predict(text1)
                calinski_harabasz_score_class5 = metrics.calinski_harabasz_score(text1, Class5)
                calinski_harabasz_score_class10 = metrics.calinski_harabasz_score(text1, Class10)
                calinski_harabasz_score_class15 = metrics.calinski_harabasz_score(text1, Class15)
                calinski_harabasz_score_class20 = metrics.calinski_harabasz_score(text1, Class20)
                if Max_calinski_harabasz_score_class5<calinski_harabasz_score_class5:
                    Max_calinski_harabasz_score_class5=calinski_harabasz_score_class5
                if Max_calinski_harabasz_score_class10<calinski_harabasz_score_class10:
                    Max_calinski_harabasz_score_class10=calinski_harabasz_score_class10
                if Max_calinski_harabasz_score_class15<calinski_harabasz_score_class15:
                    Max_calinski_harabasz_score_class15=calinski_harabasz_score_class15
                if Max_calinski_harabasz_score_class20<calinski_harabasz_score_class20:
                    Max_calinski_harabasz_score_class20=calinski_harabasz_score_class20
                sc_score_class5 = silhouette_score(text1, Class5, metric='euclidean')
                sc_score_class10 = silhouette_score(text1, Class10, metric='euclidean')
                sc_score_class15 = silhouette_score(text1, Class15, metric='euclidean')
                sc_score_class20 = silhouette_score(text1, Class20, metric='euclidean')
                if sc_score_class5 > Max_sc_score_class5:
                    Max_sc_score_class5 =sc_score_class5
                if sc_score_class10 > Max_sc_score_class10:
                    Max_sc_score_class10 =sc_score_class10
                if sc_score_class15 > Max_sc_score_class15:
                    Max_sc_score_class15 =sc_score_class15
                if sc_score_class20 > Max_sc_score_class20:
                    Max_sc_score_class20 =sc_score_class20
                ListClass5=Class5.tolist()
                ListClass10=Class10.tolist()
                ListClass15= Class15.tolist()
                ListClass20= Class20.tolist()
                ListLabel=label.tolist()
                for i in range(len(ListClass5)):
                    if ListClass5[i] == 0:
                        dictclass5['0'].append(i)
                    if ListClass5[i] == 1:
                        dictclass5['1'].append(i)
                    if ListClass5[i] == 2:
                        dictclass5['2'].append(i)
                    if ListClass5[i] == 3:
                        dictclass5['3'].append(i)
                    if ListClass5[i] == 4:
                        dictclass5['4'].append(i)
                for i in range(len(ListClass10)):
                    if ListClass10[i] == 0:
                        dictclass10['0'].append(i)
                    if ListClass10[i] == 1:
                        dictclass10['1'].append(i)
                    if ListClass10[i] == 2:
                        dictclass10['2'].append(i)
                    if ListClass10[i] == 3:
                        dictclass10['3'].append(i)
                    if ListClass10[i] == 4:
                        dictclass10['4'].append(i)
                    if ListClass10[i] == 5:
                        dictclass10['5'].append(i)
                    if ListClass10[i] == 6:
                        dictclass10['6'].append(i)
                    if ListClass10[i] == 7:
                        dictclass10['7'].append(i)
                    if ListClass10[i] == 8:
                        dictclass10['8'].append(i)
                    if ListClass10[i] == 9:
                        dictclass10['9'].append(i)
                for i in range(len(ListClass15)):
                    if ListClass15[i] == 0:
                        dictclass15['0'].append(i)
                    if ListClass15[i] == 1:
                        dictclass15['1'].append(i)
                    if ListClass15[i] == 2:
                        dictclass15['2'].append(i)
                    if ListClass15[i] == 3:
                        dictclass15['3'].append(i)
                    if ListClass15[i] == 4:
                        dictclass15['4'].append(i)
                    if ListClass15[i] == 5:
                        dictclass15['5'].append(i)
                    if ListClass15[i] == 6:
                        dictclass15['6'].append(i)
                    if ListClass15[i] == 7:
                        dictclass15['7'].append(i)
                    if ListClass15[i] == 8:
                        dictclass15['8'].append(i)
                    if ListClass15[i] == 9:
                        dictclass15['9'].append(i)
                    if ListClass15[i] == 10:
                        dictclass15['10'].append(i)
                    if ListClass15[i] == 11:
                        dictclass15['11'].append(i)
                    if ListClass15[i] == 12:
                        dictclass15['12'].append(i)
                    if ListClass15[i] == 13:
                        dictclass15['13'].append(i)
                    if ListClass15[i] == 14:
                        dictclass15['14'].append(i)
                for i in range(len(ListClass20)):
                    if ListClass20[i] == 0:
                        dictclass20['0'].append(i)
                    if ListClass20[i] == 1:
                        dictclass20['1'].append(i)
                    if ListClass20[i] == 2:
                        dictclass20['2'].append(i)
                    if ListClass20[i] == 3:
                        dictclass20['3'].append(i)
                    if ListClass20[i] == 4:
                        dictclass20['4'].append(i)
                    if ListClass20[i] == 5:
                        dictclass20['5'].append(i)
                    if ListClass20[i] == 6:
                        dictclass20['6'].append(i)
                    if ListClass20[i] == 7:
                        dictclass20['7'].append(i)
                    if ListClass20[i] == 8:
                        dictclass20['8'].append(i)
                    if ListClass20[i] == 9:
                        dictclass20['9'].append(i)
                    if ListClass20[i] == 10:
                        dictclass20['10'].append(i)
                    if ListClass20[i] == 11:
                        dictclass20['11'].append(i)
                    if ListClass20[i] == 12:
                        dictclass20['12'].append(i)
                    if ListClass20[i] == 13:
                        dictclass20['13'].append(i)
                    if ListClass20[i] == 14:
                        dictclass20['14'].append(i)
                    if ListClass20[i] == 15:
                        dictclass20['15'].append(i)
                    if ListClass20[i] == 16:
                        dictclass20['16'].append(i)
                    if ListClass20[i] == 17:
                        dictclass20['17'].append(i)
                    if ListClass20[i] == 18:
                        dictclass20['18'].append(i)
                    if ListClass20[i] == 19:
                        dictclass20['19'].append(i)
                for i in range(len(ListLabel)):
                    if ListLabel[i] == 0:
                        dictlabel['0'].append(i)
                    if ListLabel[i] == 1:
                        dictlabel['1'].append(i)
                    if ListLabel[i] == 2:
                        dictlabel['2'].append(i)
                    if ListLabel[i] == 3:
                        dictlabel['3'].append(i)
                    if ListLabel[i] == 4:
                        dictlabel['4'].append(i)
                    if ListLabel[i] == 5:
                        dictlabel['5'].append(i)
                    if ListLabel[i] == 6:
                        dictlabel['6'].append(i)
                    if ListLabel[i] == 7:
                        dictlabel['7'].append(i)
                    if ListLabel[i] == 8:
                        dictlabel['8'].append(i)
                    if ListLabel[i] == 9:
                        dictlabel['9'].append(i)
                    if ListLabel[i] == 10:
                        dictlabel['10'].append(i)
                    if ListLabel[i] == 11:
                        dictlabel['11'].append(i)
                    if ListLabel[i] == 12:
                        dictlabel['12'].append(i)
                    if ListLabel[i] == 13:
                        dictlabel['13'].append(i)
                    if ListLabel[i] == 14:
                        dictlabel['14'].append(i)
                    if ListLabel[i] == 15:
                        dictlabel['15'].append(i)
                    if ListLabel[i] == 16:
                        dictlabel['16'].append(i)
                    if ListLabel[i] == 17:
                        dictlabel['17'].append(i)
                    if ListLabel[i] == 18:
                        dictlabel['18'].append(i)
                    if ListLabel[i] == 19:
                        dictlabel['19'].append(i)
                for i in range(20):
                    dictlabel[str(i)] = set(dictlabel[str(i)])
                for i in range(5):
                    dictclass5[str(i)] = set(dictclass5[str(i)])
                for i in range(10):
                    dictclass10[str(i)] = set(dictclass10[str(i)])
                for i in range(15):
                    dictclass15[str(i)] = set(dictclass15[str(i)])
                for i in range(20):
                    dictclass20[str(i)] = set(dictclass20[str(i)])
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                KmeamsClusteringScore2 = cluster_acc(y2_pred, label)
                KmeamsClusteringScore2 =list(KmeamsClusteringScore2)
                if Kmeans_Acc < KmeamsClusteringScore2[0]:
                    Kmeans_Acc = KmeamsClusteringScore2[0]
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                clustering_accuracy= KmeamsClusteringScore2[0]
                if clustering_Max < clustering_accuracy:
                    clustering_Max = clustering_accuracy
                    Max_Acc_global_step= global_step
                KmeansAMI = metrics.adjusted_mutual_info_score(label, y2_pred)
                if Kmeans_AMI<KmeansAMI:
                    Kmeans_AMI = KmeansAMI
                MaxAmi = KmeansAMI
                if AmiMax < MaxAmi:
                    AmiMax = MaxAmi
                kmeansNmi = metrics.normalized_mutual_info_score(label, y2_pred)
                if Kmeans_Nmi < kmeansNmi:
                    Kmeans_Nmi = kmeansNmi
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                MaxNmi =  kmeansNmi
                if NmiMax < MaxNmi:
                    NmiMax = MaxNmi
                KmeansAri = metrics.normalized_mutual_info_score(label, y2_pred)
                if Kmeans_ARI < KmeansAri:
                    Kmeans_ARI = KmeansAri
                MaxAri = KmeansAri
                if AriMax < MaxAri:
                    AriMax = MaxAri
                print("calinski_harabasz_score:")
                print("class5：",Max_calinski_harabasz_score_class5,"class10：",Max_calinski_harabasz_score_class10,"class15:",Max_calinski_harabasz_score_class15,"class20:",Max_calinski_harabasz_score_class20)
                print("sc_score:")
                print("class5:",Max_sc_score_class5,"class10:",Max_sc_score_class10,"class15:",Max_sc_score_class15,"class20:",Max_sc_score_class20)

                fscore5 = Bucued.fscore(dictclass5, dictlabel, 1)
                fscore10 = Bucued.fscore(dictclass10, dictlabel, 1)
                fscore15 = Bucued.fscore(dictclass15, dictlabel, 1)
                fscore20 = Bucued.fscore(dictclass20, dictlabel, 1)
                if Max_cube5 < fscore5:
                    Max_cube5 = fscore5
                if Max_cube10 < fscore10:
                    Max_cube10 = fscore10
                if Max_cube15 < fscore15:
                    Max_cube15 = fscore15
                if Max_cube20 < fscore20:
                    Max_cube20 = fscore20
                print("class 5 Max_bcube_f1:",Max_cube5,"class 10 Max_bcube_f1",Max_cube10,"class 15 Max_bcube_f1",Max_cube15,"class 20 Max_bcube_f1",Max_cube20)
                print("Max_Acc_global_step",Max_Acc_global_step, "Kmeans_Acc",Kmeans_Acc,"Kmeans_AMI",AmiMax,"Kmeans_ARI",AriMax,"Kmeans_NMI",NmiMax)
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            if self.cfg.total_steps and self.cfg.total_steps < global_step:
                print('The total steps have been reached')
                print('Average Loss %5.3f' % (loss_sum / (i + 1)))
                if get_acc:
                    text1, label1, results = self.clustereval(get_acc, None, model)
                    total_accuracy = torch.cat(results).mean().item()
                    accuracyscore = torch.cat(results).mean().item()
                    text1 = text1[64:]
                    label = label1[64:]
                    y_pred = SpectralClustering(n_clusters=20).fit_predict(text1)
                    y2_pred = KMeans(n_clusters=20, random_state=9).fit_predict(text1)
                    ClusteringScore = cluster_acc(y_pred, label)
                    ClusteringScore2 = cluster_acc(y2_pred, label)
                    Maxscore = max(ClusteringScore[0], ClusteringScore2[0])
                    if max_acc[4] < Maxscore:
                        max_acc = total_accuracy, global_step, ClusteringScore[0],ClusteringScore2[0],Maxscore
                    print('Spectral Accuracy :', ClusteringScore[0],"K-means",ClusteringScore2[0])
                    print("Max Accuracy:", max_acc[0], "Max global_steps", max_acc[1], "Clustering", max_acc[2],"K-means",max_acc[3],"Max-clustering",max_acc[4])
                self.save(global_step)
                return
        return global_step

    def clustereval(self, evaluate, model_file, model):
        """ evaluation function """
        if model_file:
            self.model.eval()
            self.load(model_file, None)
            model = self.model.to(self.device)
            if self.cfg.data_parallel:
                model = nn.DataParallel(model)

        results = []
        iter_bar = tqdm(self.sup_iter) if model_file \
            else tqdm(deepcopy(self.eval_iter))
        text = np.zeros((128, 20))
        label = np.zeros((128))
        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad():
                logits, label1, accuracyscore, result = evaluate(model, batch)
                logits = logits.cpu().numpy()
                label1 = label1.cpu().numpy()
                text = np.concatenate((text, logits), axis=0)
                label = np.concatenate((label, label1), axis=0)

            results.append(result)
        return text, label, results

    def eval(self, evaluate, model_file, model):
        """ evaluation function """
        if model_file:
            self.model.eval()
            self.load(model_file, None)
            model = self.model.to(self.device)
            if self.cfg.data_parallel:
                model = nn.DataParallel(model)

        results = []
        iter_bar = tqdm(self.sup_iter) if model_file \
            else tqdm(deepcopy(self.eval_iter))
        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]

            with torch.no_grad():
                accuracy, result = evaluate(model, batch)
            results.append(result)

            iter_bar.set_description('Eval Acc=%5.3f' % accuracy)
        return results

    def load(self, model_file, pretrain_file):
        """ between model_file and pretrain_file, only one model will be loaded """
        if model_file:
            print('Loading the model from', model_file)
            if torch.cuda.is_available():
                self.model.load_state_dict(torch.load(model_file))
            else:
                self.model.load_state_dict(torch.load(model_file, map_location='cpu'))

        elif pretrain_file:
            print('Loading the pretrained model from', pretrain_file)
            if pretrain_file.endswith('.ckpt'):  # checkpoint file in tensorflow
                checkpoint.load_model(self.model.transformer, pretrain_file)
            elif pretrain_file.endswith('.pt'):  # pretrain model file in pytorch
                self.model.transformer.load_state_dict(
                    {key[12:]: value
                     for key, value in torch.load(pretrain_file).items()
                     if key.startswith('transformer')}
                )  # load only transformer parts

    def save(self, i):
        """ save model """
        if not os.path.isdir(os.path.join(self.cfg.results_dir, 'save')):
            os.makedirs(os.path.join(self.cfg.results_dir, 'save'))
        torch.save(self.model.state_dict(),
                   os.path.join(self.cfg.results_dir, 'save', 'model_steps_' + str(i) + '.pt'))

    def repeat_dataloader(self, iterable):
        """ repeat dataloader """
        while True:
            for x in iterable:
                yield x