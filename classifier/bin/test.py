# -*- coding: utf-8 -*-
# @Time    : 2022/5/31 16:48
# @Author  : YuJie
# @Email   : 162095214@qq.com
# @File    : predict.py
import argparse

import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from tqdm import tqdm
from transformers import BertTokenizer, AdamW

from torch import device
import torch.nn.functional as F

from classifier.data.dataset import LabeledDataset

Default_Hparams = {
    "batch_size": 16,
    "model": "../../train/model.pkl",
    "dev": "../../data/input/dev.csv",
    "device": 0,
    "max_len": 256,
    "pretrain": "../../bert-base-chinese",
}


def override_params(args):
    params = Default_Hparams
    arg_dict = vars(args)
    for key, item in arg_dict.items():
        if key in Default_Hparams.keys() and item:
            params[key] = item
    return params


def get_hparams(args):
    parser = argparse.ArgumentParser(
        description="Train a neural machine translation model.",
        usage="trainer.py [<args>] [-h | --help]"
    )

    parser.add_argument("--dev", type=str,
                        help="Path to validation file.")
    parser.add_argument("--batch_size", type=int,
                        help=" batch_size")
    parser.add_argument("--device", type=int,
                        help="device")
    parser.add_argument("--model", type=str,
                        help="device")
    parsed_args = parser.parse_args(args)

    # 更新Hparmas
    hparams = override_params(parsed_args)

    return hparams


def dev(args=None):
    hparams = get_hparams(args)

    device_index = 'cuda:' + str(hparams['device'])
    device = torch.device(device_index if torch.cuda.is_available() else 'cpu')

    test_dataset = LabeledDataset(input_path=hparams['dev'], pretrain=hparams['pretrain'], max_len=hparams['max_len'])
    test_loader = DataLoader(dataset=test_dataset, batch_size=hparams['batch_size'], shuffle=True)

    model = torch.load(hparams['model'])

    model.to(device)
    model.eval()

    predicts = []
    predict_probs = []
    with torch.no_grad():
        correct = 0
        total = 0
        for step, (input_ids, token_type_ids, attention_mask, labels) in tqdm(enumerate(test_loader)):
            input_ids, token_type_ids, attention_mask, labels = input_ids.to(device), token_type_ids.to(
                device), attention_mask.to(device), labels.to(device)

            _, predict, output,_ = model(input_ids, token_type_ids, attention_mask, labels)

            pre_numpy = predict.cpu().numpy().tolist()
            predicts.extend(pre_numpy)
            probs = F.softmax(output).detach().cpu().numpy().tolist()
            predict_probs.extend(probs)

            correct += (predict == labels).sum().item()
            total += labels.size(0)
            # print('now_predict_Accuracy : {} %'.format(100.0 * correct / total))
            # print(probs)
        res = correct / total
        print('predict_Accuracy : {} %'.format(100 * res))
        # 返回预测结果和预测的概率
        return predicts, predict_probs


if __name__ == '__main__':
    dev()
