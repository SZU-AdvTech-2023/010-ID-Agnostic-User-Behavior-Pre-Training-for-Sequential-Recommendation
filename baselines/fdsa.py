import copy
import torch
import random
from torch import nn
from recbole.model.loss import BPRLoss
from recbole.model.sequential_recommender.sasrec import SASRec

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.ac = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        x = self.ac(self.fc2(self.ac(self.fc1(x))))
        return x

class FDSA(SASRec):
    def __init__(self, config, dataset):
        super(FDSA, self).__init__(config, dataset)

        # load parameters info - SASRec backbone
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.device = config['device']
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # three train stage
        self.plm_size = config['plm_size']
        self.train_stage = config['train_stage']
        assert self.train_stage in [
            'pretrain', 'finetune_w_id', 'finetune_wo_id'
        ], f'Unknown train stage: [{self.train_stage}]'
        self.mask_ratio = config['mask_ratio']
        self.mask_token = self.n_items
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)
        self.shuffle_ratio = config['shuffle_ratio']
        self.lam = config['lam']
        self.lam_p = config['lambda']

        if self.train_stage in ['pretrain', 'finetune_wo_id']:
            self.item_embedding = None
        else:
            self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
            # for `finetune_w_id`, `item_embedding` is defined in SASRec base model

        self.plm_embedding = copy.deepcopy(dataset.mask_plm_embedding)
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)
        self.semantic_transfer_layer = MLP(config['plm_size'], config['inner_size'], config['hidden_size'])
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask

    def _neg_sample(self, item_set):
        item = random.randint(1, self.n_items - 1)
        while item in item_set:
            item = random.randint(1, self.n_items - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence

    def reconstruct_train_data(self, item_seq):
        """
        Mask item sequence for training.
        """
        device = item_seq.device
        batch_size = item_seq.size(0)

        sequence_instances = item_seq.cpu().numpy().tolist()

        # Masked Item Prediction
        # [B * Len]
        masked_item_sequence = []
        pos_items = []
        neg_items = []
        masked_index = []
        for instance in sequence_instances:
            # WE MUST USE 'copy()' HERE!
            masked_sequence = instance.copy()
            pos_item = []
            neg_item = []
            index_ids = []
            for index_id, item in enumerate(instance):
                # padding is 0, the sequence is end
                if item == 0:
                    break
                prob = random.random()
                if prob < self.mask_ratio:
                    pos_item.append(item)
                    neg_item.append(self._neg_sample(instance))
                    masked_sequence[index_id] = self.mask_token
                    index_ids.append(index_id)

            masked_item_sequence.append(masked_sequence)
            pos_items.append(self._padding_sequence(pos_item, self.mask_item_length))
            neg_items.append(self._padding_sequence(neg_item, self.mask_item_length))
            masked_index.append(self._padding_sequence(index_ids, self.mask_item_length))

        # [B Len]
        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        neg_items = torch.tensor(neg_items, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        masked_index = torch.tensor(masked_index, dtype=torch.long, device=device).view(batch_size, -1)
        return masked_item_sequence, pos_items, neg_items, masked_index

    def reconstruct_test_data(self, item_seq, item_seq_len):
        """
        Add mask token at the last position according to the lengths of item_seq
        """
        padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)  # [B]
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
        return item_seq

    def forward(self, item_seq, item_seq_len):
        # position embedding
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        # get plm embedding
        text_item_emb = self.plm_embedding(item_seq) # [2048, 50]
        text_item_emb = self.semantic_transfer_layer(text_item_emb)

        # text position add position embedding
        text_item_emb = text_item_emb + position_embedding
        if self.train_stage == 'finetune_w_id':
            text_item_emb += self.item_embedding(item_seq)

        # share layer norm and dropout
        text_item_emb = self.LayerNorm(text_item_emb)
        text_trm_input = self.dropout(text_item_emb)

        # get attention mask
        extended_attention_mask = self.get_attention_mask(item_seq)
        text_trm_output = self.trm_encoder(text_trm_input, extended_attention_mask, output_all_encoded_layers=True)
        text_output = text_trm_output[-1]
        text_output = self.gather_indexes(text_output, item_seq_len - 1)
        text_output = self.LayerNorm(text_output)
        seq_output = self.dropout(text_output)

        if self.train_stage == 'pretrain':
            # MLM modeling
            masked_item_seq, masked_pos_items, masked_neg_items, masked_index = self.reconstruct_train_data(item_seq)
            masked_position_ids = torch.arange(masked_item_seq.size(1), dtype=torch.long, device=masked_item_seq.device)
            masked_position_ids = masked_position_ids.unsqueeze(0).expand_as(masked_item_seq)
            masked_position_embedding = self.position_embedding(masked_position_ids)
            masked_text_item_emb = self.plm_embedding(masked_item_seq)
            masked_text_item_emb = self.semantic_transfer_layer(masked_text_item_emb)
            masked_input_emb = masked_text_item_emb + masked_position_embedding
            masked_input_emb = self.LayerNorm(masked_input_emb)
            masked_input_emb = self.dropout(masked_input_emb)
            masked_extended_attention_mask = self.get_attention_mask(masked_item_seq, bidirectional=True)
            masked_trm_output = self.trm_encoder(masked_input_emb, masked_extended_attention_mask,
                                                 output_all_encoded_layers=True)
            masked_feature_output = masked_trm_output[-1]

            # Permuted Item Prediction
            shuffle_item_seq = self.random_shuffle_nonzero_elements_batch(item_seq)
            permuted_item_emb = self.plm_embedding(shuffle_item_seq)
            permuted_item_emb = self.semantic_transfer_layer(permuted_item_emb)
            permuted_item_emb = permuted_item_emb + position_embedding
            permuted_input_emb = self.LayerNorm(permuted_item_emb)
            permuted_input_emb = self.dropout(permuted_input_emb)
            extended_attention_mask = self.get_attention_mask(shuffle_item_seq)
            permuted_trm_output = self.trm_encoder(
                permuted_input_emb, extended_attention_mask, output_all_encoded_layers=True
            )
            permuted_text_output = permuted_trm_output[-1]
            permuted_text_output = self.gather_indexes(permuted_text_output, item_seq_len - 1)  # [B H]
            permuted_text_output = self.LayerNorm(permuted_text_output)
            permuted_seq_output = self.dropout(permuted_text_output)

            return (seq_output, masked_feature_output, permuted_seq_output,
                    masked_item_seq, masked_pos_items, masked_neg_items, masked_index)
        else:
            return seq_output

    def random_shuffle_nonzero_elements_batch(self, tensor):
        # 获取每一行非零元素的索引
        non_zero_indices = [torch.nonzero(row).squeeze() for row in tensor]

        # 对每一行的非零元素进行随机置换
        for indices in non_zero_indices:
            # 确保至少有两个非零元素才进行置换
            # if len(indices) > 1:
            if indices.numel() > 1:
                num_elements_to_shuffle = min(int(self.shuffle_ratio * len(indices)), len(indices) - 1)
                elements_to_shuffle = indices[:num_elements_to_shuffle]
                # 随机选择要进行置换的元素
                shuffled_elements = elements_to_shuffle[torch.randperm(num_elements_to_shuffle)]
                # 仅在非零元素的位置进行 shuffle 操作
                tensor[:, elements_to_shuffle] = tensor[:, elements_to_shuffle][:, shuffled_elements.roll(1, dims=0)]

        return tensor

    def multi_hot_embed(self, masked_index, max_length):
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(masked_index.size(0), max_length, device=masked_index.device)
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def pretrain(self, interaction):
        # get item_seq
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        (seq_output, masked_seq_output, permuted_seq_output,
         masked_item_seq, masked_pos_items, masked_neg_items, masked_index) = self.forward(item_seq, item_seq_len)

        pred_index_map = self.multi_hot_embed(masked_index, masked_item_seq.size(-1))
        pred_index_map = pred_index_map.view(masked_index.size(0), masked_index.size(1), -1)
        masked_seq_output = torch.bmm(pred_index_map, masked_seq_output)

        test_item_emb = self.semantic_transfer_layer(self.plm_embedding.weight[:self.n_items])

        next_items_logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        permuted_items_logits = torch.matmul(permuted_seq_output, test_item_emb.transpose(0, 1))
        mask_items_logits = torch.matmul(masked_seq_output, test_item_emb.transpose(0, 1))

        targets = (masked_index > 0).float().view(-1)  # [B*mask_len]
        mask_item_loss = torch.sum(
            self.loss_fct(mask_items_logits.view(-1, test_item_emb.size(0)), masked_pos_items.view(-1)) * targets) \
                         / torch.sum(targets)
        next_item_loss = self.loss_fct(next_items_logits, pos_items)
        permuted_item_loss = self.loss_fct(permuted_items_logits, pos_items)
        loss = next_item_loss + mask_item_loss * self.lam + permuted_item_loss * self.lam_p
        return loss

    def calculate_loss(self, interaction):
        if self.train_stage == 'pretrain':
            return self.pretrain(interaction)

        # Loss for fine-tuning
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.semantic_transfer_layer(self.plm_embedding.weight[:self.n_items])
        if self.train_stage == 'finetune_w_id':
            test_item_emb += self.item_embedding.weight[:self.n_items]

        next_items_logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        next_item_loss = self.loss_fct(next_items_logits, pos_items)
        loss = next_item_loss
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_items_emb = self.semantic_transfer_layer(self.plm_embedding.weight[:self.n_items])
        if self.train_stage == 'finetune_w_id':
            test_items_emb += self.item_embedding.weight[:self.n_items]

        if self.train_stage == 'pretrain':
            (seq_output, masked_seq_output, permuted_seq_output,
             masked_item_seq, masked_pos_items, masked_neg_items, masked_index) = self.forward(item_seq, item_seq_len)
        else:
            seq_output = self.forward(item_seq, item_seq_len)
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores