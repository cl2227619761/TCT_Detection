import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    focal loss implementn

    loss(x, class) = -alpha*((1-softmax(x)[class])^gamma)*log(softmax(x)[class])

    args:
        alpha: 1D tensor, the scalar factor for this criterion
        gamma: float, gamma>0; reduces the relative loss for \
            well-classified examples (p > .5), putting more focus on \
            hard, misclassified examples
        size_avg: bool, by default, the losses are averaged over obs \
            for each minibatch
    """

    def __init__(self, class_nums, alpha=None, gamma=2, size_avg=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_nums, 1)

        else:
            self.alpha = alpha

        self.gamma = gamma
        self.class_nums = class_nums
        self.size_avg = size_avg

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = torch.zeros(N, C, dtype=inputs.dtype,
                                 device=inputs.device)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        batch_loss = -alpha * (torch.pow((1-probs), self.gamma)) * log_p

        if self.size_avg:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

