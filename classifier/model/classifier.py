import torch
import torch.nn as nn

import transformers


class Classifier(nn.Module):
    def __init__(self, pretrain):
        super(Classifier, self).__init__()
        # 加载预训练模型
        self.bert = transformers.BertModel.from_pretrained(pretrain)
        for param in self.bert.parameters():
            param.requires_grad = True
        # 定义线性函数
        self.dense = nn.Linear(768, 3)  # bert默认的隐藏单元数是768， 输出单元是3，表示三分类
        self.softmax = nn.Softmax(dim=-1)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        # 得到bert_output
        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # 获得预训练模型的输出
        bert_cls_hidden_state = bert_output[1]
        # 将768维的向量输入到线性层映射为二维向量
        linear_output = self.dense(bert_cls_hidden_state)
        softmax_output = self.softmax(linear_output)

        # 计算损失
        loss = self.criterion(softmax_output, labels)
        _, predict = torch.max(softmax_output.data, 1)

        return loss, predict,softmax_output
