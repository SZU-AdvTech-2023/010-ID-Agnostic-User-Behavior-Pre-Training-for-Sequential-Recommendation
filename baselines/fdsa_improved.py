import torch
import copy
import torch.nn as nn
from recbole.model.loss import BPRLoss
from recbole.model.sequential_recommender.sasrec import SASRec

class LSH(torch.nn.Module):
    def __init__(self, num_hyperplanes, maxlen, embedding_size):
        super(LSH, self).__init__()
        self.num_hyperplanes = num_hyperplanes
        self.embedding_size = embedding_size
        self.maxlen = maxlen
        self.hyperplanes = torch.nn.Parameter(torch.randn(num_hyperplanes, embedding_size))
        self.thresholds = torch.nn.Parameter(torch.rand(num_hyperplanes))

    def forward(self, vectors):
        # vectors 的形状：(batch_size, num_vectors_per_batch, embedding_size)
        vectors = vectors.view(-1, self.embedding_size)
        projections = torch.mm(vectors, self.hyperplanes.t())
        hash_values = (projections >= self.thresholds.view(1, -1)).int()

        # 返回哈希码，形状为 (batch_size, num_vectors_per_batch, num_hyperplanes)
        return hash_values.view(-1, self.maxlen, self.num_hyperplanes)

def activation_function(x):
    return x * torch.tanh(torch.log(1 + torch.exp(x)))

class StackedBlock(nn.Module):
    def __init__(self, input_dim, output_dim, layer_norm_eps):
        super(StackedBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        x = self.fc(x)
        x = activation_function(x)
        return x

class HASHDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hash_num, layer_norm_eps, num_blocks):
        super(HASHDNN, self).__init__()
        self.hash_num = hash_num
        self.hidden_size = hidden_dim
        self.layer_norm_eps = layer_norm_eps
        self.input_fc = nn.Linear(input_dim * hash_num, hidden_dim, bias=True)

        # Create a stack of blocks
        self.blocks = nn.ModuleList([StackedBlock(hidden_dim, hidden_dim, layer_norm_eps) for _ in range(num_blocks)])
        self.output_fc = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        x = self.input_fc(x)

        # Apply the stacked blocks
        for block in self.blocks:
            x = block(x)

        x = self.output_fc(x)
        return x


class LearnableGate(nn.Module):
    def __init__(self, input_size):
        super(LearnableGate, self).__init__()
        self.fc = nn.Linear(input_size * 2, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        gate = self.sigmoid(self.fc(x))
        fused_output = gate * x1 + (1 - gate) * x2

        return fused_output


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
        self.lam = config['lam']

        # multi-hash LSH
        self.num_hyperplanes = config['num_hyperplanes']
        self.hash_num = config['hash_num']
        self.dnn_num = config['dnn_num']
        self.hash_size = self.geometric_series_sum(self.num_hyperplanes)
        self.lsh_layer = torch.nn.ModuleList()
        self.hash_embs = torch.nn.ModuleList()
        self.maxlen = config['MAX_ITEM_LIST_LENGTH']
        for _ in range(self.hash_num):
            lsh = LSH(self.num_hyperplanes, self.maxlen, self.plm_size)
            self.lsh_layer.append(lsh)
            hash_embedding = nn.Embedding(self.hash_size + 1, self.hidden_size, padding_idx=0)
            self.hash_embs.append(hash_embedding)
        self.hash_dnn = HASHDNN(config['hidden_size'], config['inner_size'], config['hidden_size'],
                                self.hash_num, self.layer_norm_eps, num_blocks=self.dnn_num)
        self.gate_module = LearnableGate(input_size=config['hidden_size'])
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
        self.plm_embedding = copy.deepcopy(dataset.mask_plm_embedding)
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)
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

    def geometric_series_sum(self, n):
        return (2 ** n - 1)

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

    def forward(self, item_seq, item_seq_len):
        # position embedding
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_id_emb = self.item_embedding(item_seq)

        # get plm embedding
        text_item_emb = self.plm_embedding(item_seq)  # [2048, 50]
        text_item_hash_emb = []
        for i in range(self.hash_num):
            hash_codes = self.lsh_layer[i](text_item_emb)
            a = (2 ** torch.arange(hash_codes.size(2))).to(hash_codes.device)
            result = torch.sum((hash_codes * a), dim=2) # 哈希冲突
            code_hash_emb = self.hash_embs[i](result)
            memory_emb = torch.mul(code_hash_emb, item_id_emb) # 向量点乘 id emb 和 code emb
            text_item_hash_emb.append(memory_emb)
        text_item_hash_emb = torch.cat(text_item_hash_emb, dim=-1)
        text_item_emb = self.hash_dnn(text_item_hash_emb)
        text_item_emb = self.gate_module(item_id_emb, text_item_emb)
        text_item_emb += self.item_embedding(item_seq)
        text_item_emb += position_embedding

        # share layer norm and dropout
        text_item_emb = self.LayerNorm(text_item_emb)
        text_trm_input = self.dropout(text_item_emb)

        # get attention mask
        extended_attention_mask = self.get_attention_mask(item_seq)
        text_trm_output = self.trm_encoder(
            text_trm_input, extended_attention_mask, output_all_encoded_layers=True
        )  # [B Len H]
        text_output = text_trm_output[-1]
        text_output = self.gather_indexes(text_output, item_seq_len - 1)
        text_output = self.LayerNorm(text_output)
        seq_output = self.dropout(text_output)

        return seq_output

    def calculate_loss(self, interaction):
        # Loss for fine-tuning
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding.weight[:self.n_items]

        next_items_logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        next_item_loss = self.loss_fct(next_items_logits, pos_items)
        loss = next_item_loss
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_items_emb = self.item_embedding.weight[:self.n_items]

        seq_output = self.forward(item_seq, item_seq_len)
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores