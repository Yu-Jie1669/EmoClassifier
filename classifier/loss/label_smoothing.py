import torch.nn as nn


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.smoothing = smoothing

    def forward(self, logits, labels):
        n_label = logits.shape[-1]
        smooth_labels = labels * (1 - self.smoothing) + self.smoothing / n_label

        return self.criterion(logits, smooth_labels)
