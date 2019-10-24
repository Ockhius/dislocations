import torch


class binary_cross_entropy(torch.nn.Module):
    def __init__(self, ignore_index=250):

        super().__init__()

        self.left_bce_logits = torch.nn.BCEWithLogitsLoss()
        self.right_bce_logits = torch.nn.BCEWithLogitsLoss()

    def forward(self, seg_l, lgt, seg_r, rgt):

        _loss = 0.5 * self.left_bce_logits(seg_l, lgt) + 0.5 * self.right_bce_logits(seg_r, rgt)
        return _loss
