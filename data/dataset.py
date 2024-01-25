import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from recbole.data.dataset import SequentialDataset
import os
import numpy as np


class UniSRecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.plm_size = config['plm_size']
        self.plm_suffix = config['plm_suffix']
        plm_embedding_weight = self.load_plm_embedding()
        self.plm_embedding = self.weight2emb(plm_embedding_weight)
        self.mask_plm_embedding = self.weight2maskemb(plm_embedding_weight)

    def load_plm_embedding(self):
        feat_path = osp.join(self.config['data_path'], f'{self.dataset_name}.{self.plm_suffix}')
        tensor = torch.load(feat_path, map_location=self.config['device'])
        return tensor

    def weight2emb(self, weight):
        plm_embedding = nn.Embedding(self.item_num, self.plm_size, padding_idx=0)
        padding_embedding = torch.zeros(1, self.plm_size).to(self.config['device'])
        weight = torch.cat((padding_embedding, weight), dim=0)
        plm_embedding.weight.data = weight
        # 设置embedding.weight.requires_grad为False
        plm_embedding.weight.requires_grad = False

        return plm_embedding

    def weight2maskemb(self, weight):
        plm_embedding = nn.Embedding(self.item_num + 1, self.plm_size, padding_idx=0)
        padding_embedding = torch.zeros(1, self.plm_size).to(self.config['device'])
        weight = torch.cat((padding_embedding, weight), dim=0) # 加zero padding
        mask_embedding = torch.randn(1, self.plm_size).to(self.config['device'])
        weight = torch.cat((weight, mask_embedding), dim=0) # 加mask padding
        plm_embedding.weight.data = weight
        # 设置embedding.weight.requires_grad为False
        plm_embedding.weight.requires_grad = False

        return plm_embedding


class PretrainUniSRecDataset(UniSRecDataset):
    def __init__(self, config):
        super().__init__(config)

        self.plm_suffix_aug = config['plm_suffix_aug']
        plm_embedding_weight_aug = self.load_plm_embedding(plm_suffix_aug=self.plm_suffix_aug)
        self.plm_embedding_aug = self.weight2emb(plm_embedding_weight_aug)

    def load_plm_embedding(self, plm_suffix_aug=None):
        with open(osp.join(self.config['data_path'], f'{self.dataset_name}.pt_datasets'), 'r') as file:
            dataset_names = file.read().strip().split(',')
        self.logger.info(f'Pre-training datasets: {dataset_names}')

        d2feat = []
        for dataset_name in dataset_names:
            if plm_suffix_aug is None:
                feat_path = osp.join(self.config['data_path'], f'{dataset_name}.{self.plm_suffix}')
            else:
                feat_path = osp.join(self.config['data_path'], f'{dataset_name}.{plm_suffix_aug}')
            loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)
            d2feat.append(loaded_feat)

        iid2domain = np.zeros((self.item_num, 1))
        mapped_feat = np.zeros((self.item_num, self.plm_size))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            did, iid = token.split('-')
            loaded_feat = d2feat[int(did)]
            mapped_feat[i] = loaded_feat[int(iid)]
            iid2domain[i] = int(did)
        self.iid2domain = torch.LongTensor(iid2domain)

        return mapped_feat
