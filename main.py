# Copyright 2020 authors for paper submission:
# A Simple and Effective Usage of Self-supervised Contrastive Learning for Text Clustering,ACL-IJCNLP2021

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import models
import train
from load_data import load_data
from utils.utils import set_seeds, get_device, _get_device, torch_device_one
from utils import optim, configuration


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device=torch.device):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
        self.device = device

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        distance = torch.abs(representations.unsqueeze(1) - representations.unsqueeze(0))
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask.to(self.device) * torch.exp(similarity_matrix / self.temperature).to(
            self.device)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


class ContrastiveLossELI5(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device=torch.device, verbose=True):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose
        self.device = device

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        def l_ij(i, j):
            sim_i_j = similarity_matrix[i, j]
            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * self.batch_size,)).to(self.device).scatter_(0, torch.tensor([i]).cuda(),
                                                                                        0.0)
            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :].cuda() / self.temperature)
            )
            loss_ij = -torch.log(numerator / denominator)
            return loss_ij.squeeze(0)

        N = self.batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2 * N) * loss



def main(cfg, model_cfg):
    # Load Configuration
    cfg = configuration.params.from_json(cfg)  # Train or Eval cfg
    set_seeds(cfg.seed)
    # Load Data & Create Criterion
    data = load_data(cfg)
    if cfg.uda_mode:
        unsup_criterion = nn.KLDivLoss(reduction='none')
        data_iter = [data.sup_data_iter(), data.unsup_data_iter()] if cfg.mode == 'train' \
            else [data.sup_data_iter(), data.unsup_data_iter(), data.eval_data_iter()]  # train_eval
    else:
        data_iter = [data.sup_data_iter()]
    # using lstm model
    model = models.Lstm(20002, 300, 64, 20, 0.2, 1)
    # model = models.Classifier(model_cfg, len(data.TaskDataset.labels))
    trainer = train.Trainer(cfg, model, data_iter, optim.optim4GPU(cfg, model), get_device())

    def get_loss2(model, sup_batch, unsup_batch, global_step):

        # input_ids, segment_ids, input_mask, Aug_input_ids, Agu_segment_ids, Agu_input_mask = sup_batch
        input_ids, agu_input_ids = sup_batch
        sup_size = input_ids.shape[0]
        unsup_size = None
        input_ids = torch.cat((input_ids, agu_input_ids), dim=0)
        if unsup_batch:
            '''
            ori_input_ids, ori_segment_ids, ori_input_mask, \
            aug_input_ids, aug_segment_ids, aug_input_mask = unsup_batch
            unsup_size= ori_input_ids.shape[0]
            input_ids = torch.cat((input_ids,ori_input_ids, aug_input_ids), dim=0)
            segment_ids = torch.cat((segment_ids,ori_segment_ids, aug_segment_ids), dim=0)
            input_mask = torch.cat((input_mask,ori_input_mask,aug_input_mask), dim=0)
            '''
            ori_input_ids, aug_input_ids = unsup_batch
            unsup_size = ori_input_ids.shape[0]
            # print(input_ids.shape, ori_input_ids.shape, aug_input_ids.shape)
            input_ids = torch.cat((input_ids, ori_input_ids, aug_input_ids), dim=0)
        # logits_sup = model(input_ids, segment_ids, input_mask)
        logits_sup = model(input_ids)
        text1, text2 = logits_sup[:sup_size], logits_sup[sup_size:2 * sup_size]
        contrassloss = ContrastiveLoss(sup_size, 0.5, get_device())
        contrassloss = contrassloss(text1, text2)
        sup_loss = contrassloss
        if unsup_batch:
            with torch.no_grad():
                ori_logits = logits_sup[2 * sup_size:2 * sup_size + unsup_size]
                ori_prob = F.softmax(ori_logits, dim=-1)  # KLdiv target
                unsup_loss_mask = torch.ones((unsup_size), dtype=torch.float32)
                unsup_loss_mask = unsup_loss_mask.to(_get_device())
                uda_softmax_temp = cfg.uda_softmax_temp if cfg.uda_softmax_temp > 0 else 1.
                aug_log_prob = F.log_softmax(logits_sup[2 * sup_size + unsup_size:] / uda_softmax_temp, dim=-1)

            if aug_log_prob.shape != ori_prob.shape:
                return sup_loss, None, None
            else:
                unsup_loss = torch.sum(unsup_criterion(aug_log_prob, ori_prob), dim=-1)
                unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(
                    torch.sum(unsup_loss_mask, dim=-1),
                    torch_device_one())
                if unsup_loss.cpu().numpy() > 0.1:
                    final_loss = sup_loss + cfg.uda_coeff * unsup_loss
                else:
                    final_loss = sup_loss + cfg.uda_coeff * unsup_loss
                return final_loss, sup_loss, unsup_loss
        return sup_loss, None, None

    # evaluation
    def get_acc(model, batch):
        # input_ids, segment_ids, input_mask, label_id, sentence = batch
        # input_ids, segment_ids, input_mask, label_id = batch
        input_ids, label_id = batch
        logits = model(input_ids)
        _, label_pred = logits.max(1)

        result = (label_pred == label_id).float()
        accuracy = result.mean()
        return logits, label_id, accuracy, result

    if cfg.mode == 'train':
        trainer.train(get_loss2, get_acc, cfg.model_file, cfg.pretrain_file)

    if cfg.mode == 'train_eval':
        trainer.train(get_loss2, get_acc, cfg.model_file, cfg.pretrain_file)


if __name__ == '__main__':
    main('config/uda.json', 'config/lstm.json')
