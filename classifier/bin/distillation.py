import textbrewer
import torch
from textbrewer import GeneralDistiller
from textbrewer import TrainingConfig, DistillationConfig

import argparse
import datetime

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

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
    "smoothing": 0.1,
    "teacher": "../../train/model-ls.pkl"
}


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


def distillation(args=None):
    hparams = get_hparams(args)

    device_index = 'cuda:' + str(hparams['device'])
    device = torch.device(device_index if torch.cuda.is_available() else 'cpu')

    train_loader, dev_loader, vocab_size = get_dataloader(hparams)
    print("data successfully")

    teacher_model = torch.load(hparams['teacher'], map_location=device)

    student_model = Classifier(hparams['pretrain'], mlm_train=hparams['mlm_train'], vocab_size=vocab_size,
                               smoothing=hparams['smoothing'])

    teacher_model.to(device=device)
    student_model.to(device=device)

    print("\nteacher_model's parametrers:")
    result, _ = textbrewer.utils.display_parameters(teacher_model, max_level=3)
    print(result)

    print("student_model's parametrers:")
    result, _ = textbrewer.utils.display_parameters(student_model, max_level=3)
    print(result)

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

    import torch.nn.functional as F

    # Define callback function
    def predict(model, eval_dataloader, step, device):
        model.to(device)
        model.eval()

        predicts = []
        predict_probs = []
        with torch.no_grad():
            correct = 0
            total = 0
            for _step, (input_ids, token_type_ids, attention_mask, labels) in enumerate(eval_dataloader):
                input_ids, token_type_ids, attention_mask, labels = input_ids.to(device), token_type_ids.to(
                    device), attention_mask.to(device), labels.to(device)

                _, predict, output, _ = model(input_ids, token_type_ids, attention_mask, labels)

                pre_numpy = predict.cpu().numpy().tolist()
                predicts.extend(pre_numpy)
                probs = F.softmax(output).detach().cpu().numpy().tolist()
                predict_probs.extend(probs)

                correct += (predict == labels).sum().item()
                total += labels.size(0)
                # print('now_predict_Accuracy : {} %'.format(100.0 * correct / total))
                # print(probs)
            res = correct / total
            print('step = {} predict_Accuracy = {} %'.format(step, 100 * res))

    from functools import partial

    callback_fun = partial(predict, eval_dataloader=dev_loader, device=device)  # fill other arguments

    def simple_adaptor(batch, model_outputs):
        # The second element of model_outputs is the logits before softmax
        # The third element of model_outputs is hidden states
        return {'logits': model_outputs[2],
                'hidden': model_outputs[3].hidden_states,
                'attention': model_outputs[3].attentions,
                'inputs_mask': batch[2]}

    train_config = TrainingConfig(device=device, ckpt_frequency=5, output_dir="../../distillation/")
    distill_config = DistillationConfig(
        temperature=8,
        hard_label_weight=0,
        kd_loss_type='ce',
        probability_shift=False,
        intermediate_matches=[
            {'layer_T': 2, 'layer_S': 0, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1},
            {'layer_T': 5, 'layer_S': 1, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1},
            {'layer_T': 8, 'layer_S': 2, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1},

            {'layer_T': 2, 'layer_S': 0, 'feature': 'attention', 'loss': 'attention_mse', 'weight': 1},
            {'layer_T': 5, 'layer_S': 1, 'feature': 'attention', 'loss': 'attention_mse', 'weight': 1},
            {'layer_T': 8, 'layer_S': 2, 'feature': 'attention', 'loss': 'attention_mse', 'weight': 1}
        ]
    )

    print("train_config:")
    print(train_config)

    print("distill_config:")
    print(distill_config)

    distiller = GeneralDistiller(
        train_config=train_config, distill_config=distill_config,
        model_T=teacher_model, model_S=student_model,
        adaptor_T=simple_adaptor, adaptor_S=simple_adaptor)

    epochs = hparams['epoch']
    # arguments dict except 'optimizer'
    scheduler_args = {'num_warmup_steps': int(0.1 * epochs * len(train_loader)),
                      'num_training_steps': epochs * len(train_loader)}

    # Start distilling
    with distiller:
        distiller.train(optimizer, train_loader, num_epochs=epochs,
                        scheduler_class=get_linear_schedule_with_warmup, scheduler_args=scheduler_args,
                        callback=callback_fun)


if __name__ == '__main__':
    distillation()
