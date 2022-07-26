import random
from typing import Union

import pandas as pd
import six
import torch
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data.dataset import T_co
from transformers import BertTokenizer, AdamW


class Vocabulary:
    def __init__(self, filename):
        self._idx2word = {}
        self._word2idx = {}
        cnt = 0

        with open(filename, "rb") as fd:
            for line in fd:
                self._word2idx[line.strip()] = cnt
                self._idx2word[cnt] = line.strip()
                cnt = cnt + 1

    def __getitem__(self, key: Union[bytes, int]):
        if isinstance(key, int):
            return self._idx2word[key]
        elif isinstance(key, bytes):
            return self._word2idx[key]
        elif isinstance(key, str):
            key = key.encode("utf-8")
            return self._word2idx[key]
        else:
            raise LookupError("Cannot lookup key %s." % key)

    def __contains__(self, key):
        if isinstance(key, str):
            key = key.encode("utf-8")

        return key in self._word2idx

    def __iter__(self):
        return six.iterkeys(self._word2idx)

    def __len__(self):
        return len(self._idx2word)


def encode(max_len, pretrain, text_list):
    # 将text_list embedding成bert模型可用的输入形式
    # 加载分词模型
    tokenizer = BertTokenizer.from_pretrained(pretrain)
    tokenizer = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='pt'  # 返回的类型为pytorch tensor
    )
    input_ids = tokenizer['input_ids']
    token_type_ids = tokenizer['token_type_ids']
    attention_mask = tokenizer['attention_mask']
    return input_ids, token_type_ids, attention_mask


class LabeledDataset(Dataset):
    def __init__(self, input_path, pretrain, max_len):
        f = open(input_path, encoding="GB2312", errors='ignore')
        df = pd.read_csv(f)
        f.close()

        text_list = list(df['微博中文内容'].astype('str'))
        labels = list(df['情感倾向'].astype('int'))

        self.input_ids, self.token_type_ids, self.attention_mask = encode(max_len=max_len,
                                                                          pretrain=pretrain,
                                                                          text_list=text_list)
        self.labels = torch.tensor(labels)
        self.labels += 1

        self.vocab = Vocabulary(filename=pretrain + "/vocab.txt")

        # mlm_data [(mlm_ids,pred_pos_s,labels),(),()]
        # self.mlm_data = [self.get_mlm_data(example) for example in self.input_ids]

    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_mask[index], self.labels[index], \
               # self.mlm_data[index]

    def __len__(self):
        return self.input_ids.shape[0]

    @staticmethod
    def replace_mlm_tokens(origin_ids, candidate_pred_pos, num_mlm_pred,
                           vocab: Vocabulary):

        # 为遮蔽语言模型的输入创建新的词元副本
        mlm_ids = [id for id in origin_ids]
        positions_and_labels = []
        mask_id = vocab[b'[MASK]']

        random.shuffle(candidate_pred_pos)
        for pos in candidate_pred_pos:
            if len(positions_and_labels) >= num_mlm_pred:
                break
            replace_id = None
            # 80%将词替换为“[MASK]”
            if random.random() < 0.8:
                replace_id = mask_id
            else:
                # 10%保持词不变
                if random.random() < 0.5:
                    replace_id = origin_ids[pos]
                # 10%用随机词替换该词
                else:
                    replace_id = random.randint(0, len(vocab) - 1)
            mlm_ids[pos] = replace_id
            positions_and_labels.append((pos, origin_ids[pos]))
        return mlm_ids, positions_and_labels

    def get_mlm_data(self, input_ids):

        candidate_pred_positions = []

        cls_id = self.vocab[b'[CLS]']
        sep_id = self.vocab[b'[SEP]']

        for i, token in enumerate(input_ids):
            if token in [cls_id, sep_id]:
                continue
            candidate_pred_positions.append(i)

        # 15%的随机词元
        num_mlm_preds = max(1, round(len(candidate_pred_positions) * 0.15))

        mlm_input_ids, pred_positions_and_labels = self.replace_mlm_tokens(
            input_ids, candidate_pred_positions, num_mlm_preds, self.vocab)

        pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
        pred_positions = [v[0] for v in pred_positions_and_labels]

        # label是id
        mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
        return mlm_input_ids, pred_positions, mlm_pred_labels

    def get_vocab_size(self):
        return len(self.vocab)


class UnlabeledDataset(Dataset):

    def __init__(self, input_path, pretrain, max_len):
        f = open(input_path, encoding="UTF-8", errors='ignore')
        df = pd.read_csv(f)
        f.close()
        text_list = list(df['微博中文内容'].astype('str'))

        self.input_ids, self.token_type_ids, self.attention_mask = encode(max_len=max_len,
                                                                          pretrain=pretrain,
                                                                          text_list=text_list)

        self.vocab = Vocabulary(filename=pretrain + "/vocab.txt")

        # mlm_data [(mlm_ids,pred_pos_s,labels),(),()]
        self.mlm_data = [self.get_mlm_data(example) for example in self.input_ids]

    def __getitem__(self, index):
        return self.mlm_data[index]

    def __len__(self):
        return len(self.mlm_data)

    @staticmethod
    def replace_mlm_tokens(origin_ids, candidate_pred_pos, num_mlm_pred,
                           vocab: Vocabulary):

        # 为遮蔽语言模型的输入创建新的词元副本
        mlm_ids = [id for id in origin_ids]
        positions_and_labels = []
        mask_id = vocab[b'[MASK]']

        random.shuffle(candidate_pred_pos)
        for pos in candidate_pred_pos:
            if len(positions_and_labels) >= num_mlm_pred:
                break
            replace_id = None
            # 80%将词替换为“[MASK]”
            if random.random() < 0.8:
                replace_id = mask_id
            else:
                # 10%保持词不变
                if random.random() < 0.5:
                    replace_id = origin_ids[pos]
                # 10%用随机词替换该词
                else:
                    replace_id = random.randint(0, len(vocab) - 1)
            mlm_ids[pos] = replace_id
            positions_and_labels.append((pos, origin_ids[pos]))
        return mlm_ids, positions_and_labels

    def get_mlm_data(self, input_ids):

        candidate_pred_positions = []

        cls_id = self.vocab[b'[CLS]']
        sep_id = self.vocab[b'[SEP]']

        for i, token in enumerate(input_ids):
            if token in [cls_id, sep_id]:
                continue
            candidate_pred_positions.append(i)

        # 15%的随机词元
        num_mlm_preds = max(1, round(len(candidate_pred_positions) * 0.15))

        mlm_input_ids, pred_positions_and_labels = self.replace_mlm_tokens(
            input_ids, candidate_pred_positions, num_mlm_preds, self.vocab)

        pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
        pred_positions = [v[0] for v in pred_positions_and_labels]

        # label是id
        mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
        return mlm_input_ids, pred_positions, mlm_pred_labels
