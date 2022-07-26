import textbrewer
import torch
from textbrewer import GeneralDistiller
from textbrewer import TrainingConfig, DistillationConfig

import argparse
import datetime

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from transformers import AdamW

from classifier.data.dataset import LabeledDataset
from torch.utils.data import DataLoader

from classifier.model.classifier import Classifier
from torch import device

from classifier.model.classifier import Classifier

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

    # to dataloader
    # TODO 数据平衡？
    train_loader = DataLoader(dataset=train_dataset, batch_size=hparams['batch_size'], shuffle=True)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=hparams['batch_size'], shuffle=True)

    return train_loader, dev_loader, vocab_size


def main(args=None):
    hparams = get_hparams(args)
    print("get params successfully")

    train_loader, dev_loader, vocab_size = get_dataloader(hparams)

    # Show the statistics of model parameters
    teacher_model = torch.load("../../train/model.pkl")
    print("\nteacher_model's parametrers:")
    result, _ = textbrewer.utils.display_parameters(teacher_model, max_level=3)
    print(result)

    student_model = Classifier(hparams['pretrain'], mlm_train=hparams['mlm_train'], vocab_size=vocab_size,
                               smoothing=hparams['smoothing'])
    print("student_model's parametrers:")
    result, _ = textbrewer.utils.display_parameters(student_model, max_level=3)
    print(result)

    # Define an adaptor for interpreting the model inputs and outputs
    def simple_adaptor(batch, model_outputs):
        # The second and third elements of model outputs are the logits and hidden states
        return {'logits': model_outputs[1],
                'hidden': model_outputs[2]}

    # Training configuration
    train_config = TrainingConfig()
    # Distillation configuration
    # Matching different layers of the student and the teacher
    # We match 0-0 and 8-2 here for demonstration
    distill_config = DistillationConfig(
        intermediate_matches=[
            {'layer_T': 0, 'layer_S': 0, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1},
            {'layer_T': 1, 'layer_S': 1, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1}])

    # Build distiller
    distiller = GeneralDistiller(
        train_config=train_config, distill_config=distill_config,
        model_T=teacher_model, model_S=student_model,
        adaptor_T=simple_adaptor, adaptor_S=simple_adaptor)

    param_optimizer = list(student_model.named_parameters())
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

    # Start!
    with distiller:
        distiller.train(optimizer, train_loader, num_epochs=1,scheduler=scheduler,
                        callback=None)

if __name__ == '__main__':
    main()