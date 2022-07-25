import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset
from transformers import BertTokenizer, AdamW


class LabeledDataset(Dataset):
    def __init__(self, input_path, pretrain, max_len):
        f = open(input_path, encoding="GB2312", errors='ignore')
        df = pd.read_csv(f)
        f.close()

        text_list = list(df['微博中文内容'].astype('str'))
        labels = list(df['情感倾向'].astype('int'))

        # 调用encoder函数，获得预训练模型的三种输入形式
        self.input_ids, self.token_type_ids, self.attention_mask = self.encode(max_len=max_len,
                                                                               pretrain=pretrain,
                                                                               text_list=text_list)
        self.labels = torch.tensor(labels)
        self.labels += 1

    def __getitem__(self, idx):
        return self.input_ids[idx], self.token_type_ids[idx], self.attention_mask[idx], self.labels[idx]

    def __len__(self):
        return self.input_ids.shape[0]

    @staticmethod
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


