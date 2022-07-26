import torch.nn as nn
import torch.nn.functional as F


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss



class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self,reduction, smoothing=0.0):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.smoothing = smoothing
        self.reduction=reduction

    def forward(self, logits, target):
        n = logits.size()[-1]
        log_preds = F.log_softmax(logits, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)

