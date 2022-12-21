import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    '''
        Non weighted version of Focal Loss
    '''
    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # alpha parameter to balance class weights
        self.gamma = gamma   # higher the value of γ, the lower the loss for well-classified examples,
        # so we could turn the attention of the model more towards ‘hard-to-classify examples.
        # Having higher γ extends the range in which an example receives low loss.

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='mean')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        return F_loss
