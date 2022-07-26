import torch
import torch.nn as nn
from classifier.loss.label_smoothing import LabelSmoothingCrossEntropy

import transformers


class MaskLM(nn.Module):

    def __init__(self, vocab_size, num_inputs=768, num_hiddens=768):
        super(MaskLM, self).__init__()

        # 参数：input_size,hidden_size,output_size（词表大小）,dropout*
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, x, pred_positions):
        """
        Args:
            x: input
            pred_positions: [batch_size,num_pred]

        Returns:
        """
        num_pred = pred_positions.shape[1]
        # [batch_size,num_pred] -> [batch_size*num_pred]
        pred_positions = pred_positions.reshape(-1)

        batch_size = x.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # ->[0,0,0...1,1,1...,2,2,2...](num_pred*batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred)

        # -> [batch0pos0,batch0pos1,batch0pos2,...batch1pos0...]
        masked_input = x[batch_idx, pred_positions]
        # ->[batch_size,num_pred,1]
        masked_input = masked_input.reshape((batch_size, num_pred, -1))

        mlm = self.mlp(masked_input)
        return mlm


class Classifier(nn.Module):
    def __init__(self, pretrain, vocab_size, mlm_train=False, smoothing=None):
        super(Classifier, self).__init__()
        # 加载预训练模型
        self.bert = transformers.BertModel.from_pretrained(pretrain)
        for param in self.bert.parameters():
            param.requires_grad = True
        # 定义线性函数
        self.dense = nn.Linear(768, 3)  # bert默认的隐藏单元数是768， 输出单元是3，表示三分类
        self.softmax = nn.Softmax(dim=-1)

        self.mlm_train = mlm_train
        if self.mlm_train:
            self.mlm = MaskLM(vocab_size=vocab_size)

        if smoothing:
            self.criterion = LabelSmoothingCrossEntropy(smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, labels):

        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        bert_cls_hidden_state = bert_output[1]

        linear_output = self.dense(bert_cls_hidden_state)
        softmax_output = self.softmax(linear_output)

        # 计算损失
        loss = self.criterion(softmax_output, labels)
        _, predict = torch.max(softmax_output.data, 1)

        return loss, predict, softmax_output
