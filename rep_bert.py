'''

今天任务：采用bert-cased模型提取数据集中title的语义表示，并存储下来，直接使用

'''
from transformers import BertModel, BertTokenizer, BertConfig
import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

def read_news_bert(datatset_text, datatset_item2index):

    # 读取文件
    text_file = pd.read_csv(datatset_text, sep='\t', header=0)
    item2index = pd.read_csv(datatset_item2index, sep='\t',header=None, names=['item_id:token','id:token'])
    data_merged = pd.merge(text_file, item2index, on="item_id:token", how="left")
    data_merged = data_merged.sort_values(by='id:token')

    return data_merged

word_embedding_dim = 768
num_words_title = 50
device ='cuda:3'

bert_model_load = './bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(bert_model_load)
config = BertConfig.from_pretrained(bert_model_load, output_hidden_states=True)
bert_model = BertModel.from_pretrained(bert_model_load, config=config).to(device)
# print(bert_model)

# fix model param
for index, (name, param) in enumerate(bert_model.named_parameters()):
    param.requires_grad = False
bert_model.eval()

original_path = './dataset/text_data/'

dataset_list = ['Food'] # , 'Pantry', 'Arts'

for dataset_name in dataset_list:
    datatset_text = original_path + dataset_name + '/' + dataset_name + '.text'
    datatset_item2index = original_path  + dataset_name + '/' + dataset_name + '.item2index'
    # datatset_text = './dataset/text_data/Pantry/Pantry.text'
    # datatset_item2index = './dataset/text_data/Pantry/Pantry.item2index'
    saved_pth_name = original_path + '/' + dataset_name + '/' + dataset_name + '_text.pth'
    data_merged = read_news_bert(datatset_text, datatset_item2index)

    i = 1
    item_word_embs = []
    for index, row in data_merged.iterrows():
        # item_id = row['id:token']
        # doc_name = row['item_id:token']
        title = row['text:token_seq']
        batch_tokenized = tokenizer.encode_plus(title, max_length=num_words_title, padding='max_length', truncation=True) # add_special_tokens=True, return_attention_mask=True
        with torch.no_grad():
            input_ids = torch.tensor(batch_tokenized['input_ids']).unsqueeze(0).to(device)
            token_type_ids = torch.tensor(batch_tokenized['token_type_ids']).unsqueeze(0).to(device)
            attention_mask = torch.tensor(batch_tokenized['attention_mask']).unsqueeze(0).to(device)
            bert_output = bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            cls_vector = bert_output.last_hidden_state[:, 0, :]
            # print(cls_vector)
            # print(cls_vector.shape())
            print(i)
            i += 1
            item_word_embs.append(cls_vector)

    final_vectors = torch.stack(tensors=item_word_embs, dim=0)
    # print(final_vectors)
    # print(final_vectors.shape)
    final_vectors = final_vectors.squeeze(1)
    # print(final_vectors.shape)
    torch.save(final_vectors, saved_pth_name)
    print('finish saved' + dataset_name)