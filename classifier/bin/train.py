import argparse
import datetime

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from transformers import AdamW

from classifier.data.dataset import LabeledDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from classifier.model.classifier import Classifier
from torch import device

Default_Hparams = {
    "batch_size": 2,
    "input": "../../data/input/train.csv",
    "dev": "../../data/input/dev.csv",
    "output": "../../train/model.pkl",
    "lr": 1e-5,  # learning_rate
    "epoch": 2,
    'weight_decay': 0.01,
    "device": 0,
    "max_len": 256,
    "pretrain": "../../bert-base-chinese",
    "mlm_train": False,
    "smoothing": None,
    "temperature": 1.0,
    "checkpoint": None,
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

    parser.add_argument("--input", type=str, help="Path to source and target corpus.")
    parser.add_argument("--output", type=str, help="Path to load/store checkpoints.")
    parser.add_argument("--dev", type=str, help="Path to validation file.")
    parser.add_argument("--batch_size", type=int, help=" batch_size")
    parser.add_argument("--lr", type=float, help="Path to pre-trained checkpoint.")
    parser.add_argument("--epoch", type=int, help="epoch for train")
    parser.add_argument("--device", type=int, help="device")
    parser.add_argument("--mlm_train", type=bool, help="True to train mlm task")
    parser.add_argument("--smoothing", type=float, help="label smoothing loss")
    parser.add_argument("--temperature", type=float, help="sample temperature")
    parser.add_argument("--checkpoint", type=str, help="load checkpoint model")

    parsed_args = parser.parse_args(args)

    # 更新Hparmas
    hparams = override_params(parsed_args)

    return hparams


def get_dataloader(hparams):
    # labeled dataset
    train_dataset = LabeledDataset(input_path=hparams['input'], pretrain=hparams['pretrain'],
                                   max_len=hparams['max_len'])
    dev_dataset = LabeledDataset(input_path=hparams['dev'], pretrain=hparams['pretrain'], max_len=hparams['max_len'])

    vocab_size = train_dataset.get_vocab_size()

    if hparams['temperature']:
        temperature = hparams['temperature']
        weight_list = train_dataset.get_weight_list()
        temp_list = [cnt ** (1.0 / temperature) for cnt in weight_list]
        weight = [temp_list[int(label)] * 1.0 / sum(temp_list) for _, _, _, label in train_dataset]
        sampler = WeightedRandomSampler(weight, num_samples=len(train_dataset))
        train_loader = DataLoader(dataset=train_dataset, batch_size=hparams['batch_size'], shuffle=False,
                                  sampler=sampler)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=hparams['batch_size'], shuffle=True)

    dev_loader = DataLoader(dataset=dev_dataset, batch_size=hparams['batch_size'], shuffle=True)

    return train_loader, dev_loader, vocab_size


def dev(model, dev_loader, device):
    # 将模型放到服务器上
    model.to(device)
    # 设定模式为验证模式
    model.eval()
    # 设定不会有梯度的改变仅作验证
    with torch.no_grad():
        correct = 0
        total = 0
        for step, (input_ids, token_type_ids, attention_mask, labels) in tqdm(enumerate(dev_loader),
                                                                              desc='Dev Itreation:'):
            print("Dev Step[{}/{}]".format(step + 1, len(dev_loader)))
            input_ids, token_type_ids, attention_mask, labels = input_ids.to(device), token_type_ids.to(
                device), attention_mask.to(device), labels.to(device)
            loss, predict, _, _ = model(input_ids, token_type_ids, attention_mask, labels)
            correct += (predict == labels).sum().item()
            total += labels.size(0)
        res = correct / total
        return res


def train(args=None):
    hparams = get_hparams(args)
    print("get params successfully")

    device_index = 'cuda:' + str(hparams['device'])
    device = torch.device(device_index if torch.cuda.is_available() else 'cpu')

    train_loader, dev_loader, vocab_size = get_dataloader(hparams)
    print("get data successfully")

    model = Classifier(hparams['pretrain'], mlm_train=hparams['mlm_train'], vocab_size=vocab_size,
                       smoothing=hparams['smoothing'])

    if hparams['checkpoint']:
        checkpoint = torch.load(hparams['checkpoint'], map_location=device)
        model.load_state_dict(checkpoint)

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # 设置模型参数的权重衰减
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': hparams['weight_decay']},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # 学习率的设置
    optimizer_params = {'lr': hparams['lr'], 'eps': 1e-6, 'correct_bias': False}
    # 使用AdamW 主流优化器
    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_params)
    # 学习率调整器，检测准确率的状态，然后衰减学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-7, patience=5, verbose=True,
                                  threshold=0.0001, eps=1e-08)

    epochs = hparams['epoch']
    best_acc = 0
    correct = 0
    total = 0
    print('Training and verification begin!')

    for epoch in range(epochs):
        for step, (input_ids, token_type_ids, attention_mask, labels) in enumerate(train_loader):
            model.train()
            # 从实例化的DataLoader中取出数据，并通过 .to(device)将数据部署到服务器上
            input_ids, token_type_ids, attention_mask, labels = input_ids.to(device), token_type_ids.to(
                device), attention_mask.to(device), labels.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 将数据输入到模型中获得输出
            loss, predict, _, _ = model(input_ids, token_type_ids, attention_mask, labels)

            correct += (predict == labels).sum().item()
            total += labels.size(0)
            loss.backward()
            optimizer.step()
            # 每两步进行一次打印
            if (step + 1) % 2 == 0:
                train_acc = correct / total
                print("[Train] {} Epoch[{}/{}],step[{}/{}],tra_acc={:.6f}%,loss={:.6f}".format(
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1, epochs,
                                                                           step + 1, len(train_loader),
                                                                           train_acc * 100, loss.item()))
            # 每五十次进行一次验证
            if (step + 1) % 50 == 0:
                train_acc = correct / total
                # 调用验证函数dev对模型进行验证，并将有效果提升的模型进行保存
                acc = dev(model, dev_loader, device)
                if best_acc < acc:
                    best_acc = acc
                    # 模型保存路径
                    path = hparams['output']
                    torch.save(model, path)
                print(
                    "[DEV] {} Epoch[{}/{}],step[{}/{}],tra_acc={:.6f} %,bestAcc={:.6f}%,dev_acc={:.6f} %,loss={:.6f}".format(
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1, epochs, step + 1,
                        len(train_loader), train_acc * 100, best_acc * 100, acc * 100,
                        loss.item()))
        scheduler.step(best_acc)


if __name__ == '__main__':
    train()
