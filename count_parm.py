import torch
from torch import nn
from torch.nn import Parameter
import argparse
import torch
from recbole.config import Config
from recbole.data import data_preparation
from baselines.fdsa import FDSA
from data.dataset import UniSRecDataset

def count_parameters_by_grad_status(model):
    total_params = 0
    grad_params = 0
    no_grad_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_params += param.numel()
        else:
            no_grad_params += param.numel()

        total_params += param.numel()

    return grad_params, no_grad_params, total_params

def count_parameters_by_module(model, module_name):
    total_params = 0

    for name, param in model.named_parameters():
        if module_name in name:
            total_params += param.numel()

    return total_params

def count_parameters(model):
    total_params = 0
    id_embedding_params = 0
    text_embedding_params = 0
    trmsformer_parms = 0
    semantic_transfer_params = 0
    position_emb_params = 0
    other_parms = 0

    for name, param in model.named_parameters():
        if "item_embedding" in name:
            id_embedding_params = param.numel()
        elif 'plm_embedding' in name:
            text_embedding_params = param.numel()
        total_params += param.numel()

    position_emb_params = count_parameters_by_module(model, 'position_embedding')
    trmsformer_parms = count_parameters_by_module(model, 'trm_encoder')
    semantic_transfer_params = count_parameters_by_module(model, 'semantic_transfer_layer')
    other_parms = total_params - id_embedding_params - text_embedding_params - semantic_transfer_params - trmsformer_parms - position_emb_params
    print(f"id_embedding_params: {id_embedding_params}, 占比：{id_embedding_params / total_params * 100:.2f}%")
    print(f"text_embedding_params: {text_embedding_params}, 占比：{text_embedding_params / total_params * 100:.2f}%")
    print(f"semantic_transfer_params: {semantic_transfer_params}, 占比：{semantic_transfer_params / total_params * 100:.2f}%")
    print(f"trmsformer_parms: {trmsformer_parms}, 占比：{trmsformer_parms / total_params * 100:.2f}%")
    print(f"position_emb_params: {position_emb_params}, 占比：{position_emb_params / total_params * 100:.2f}%")
    print(f"other_parms: {other_parms}, 占比：{other_parms / total_params * 100:.2f}%")


def log_model(dataset, pretrained_file, **kwargs):
    props = ['props/FDSA.yaml', 'props/finetune.yaml']
    config = Config(model=FDSA, dataset=dataset, config_file_list=props, config_dict=kwargs)
    dataset = UniSRecDataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = FDSA(config, train_data.dataset).to(config['device'])

    # Load pre-trained model
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    grad_params, no_grad_params, total_params = count_parameters_by_grad_status(model)

    print(f"可训练参数数量: {grad_params}")
    print(f"不可训练参数数量: {no_grad_params}")
    print(f"总参数数量: {total_params}")
    print(f"可训练参数占比: {grad_params / total_params * 100:.2f}%")
    print(f"不可训练参数占比: {no_grad_params / total_params * 100:.2f}%")

    count_parameters(model)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='Instruments', help='dataset name')
    parser.add_argument('-p', type=str, default='saved/FDSA-Dec-15-2023_16-56-33.pth', help='pre-trained model path')
    parser.add_argument('-m', type=str, default='FDSA', help='name of models')
    args, unparsed = parser.parse_known_args()
    print(args)

    log_model(args.d, pretrained_file=args.p)


