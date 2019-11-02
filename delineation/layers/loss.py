import torch
import torch.nn.functional as F

class binary_cross_entropy(torch.nn.Module):
    def __init__(self, ignore_index=250):

        super().__init__()

        self.left_bce_logits = torch.nn.BCEWithLogitsLoss()
        self.right_bce_logits = torch.nn.BCEWithLogitsLoss()

    def forward(self, seg_l, lgt, seg_r, rgt):

        _loss = 0.5 * self.left_bce_logits(seg_l, lgt) + 0.5 * self.right_bce_logits(seg_r, rgt)
        return _loss


class smooth_l1_masked_dispartiy(torch.nn.Module):
    def __init__(self):

        super().__init__()

    def forward(self, dl, dlgt, lgt):

        mask = lgt > 0
        _loss = F.smooth_l1_loss(dl[mask], dlgt.unsqueeze(1)[mask], reduction='mean')
        _loss[torch.isnan(_loss)] = 0
        return _loss
