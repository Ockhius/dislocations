import torch
import torch.nn.functional as F
from delineation.utils import *

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


class smooth_l1_disparity_and_edge_warp(torch.nn.Module):
    def __init__(self):

        super().__init__()

        self.bce_logits = torch.nn.BCEWithLogitsLoss()

    def forward(self, dl, seg_r, dlgt, lgt):

        mask = lgt > 0
        recon_l = warp(seg_r, dl)

        _loss_recon = self.bce_logits(recon_l[mask], lgt[mask])
        _loss_recon[torch.isnan(_loss_recon)] = 0

        _loss = F.smooth_l1_loss(dl[mask], dlgt.unsqueeze(1)[mask], reduction='mean')
        _loss[torch.isnan(_loss)] = 0

        return _loss + 0.5 * _loss_recon, _loss


class smooth_l1_disparity_and_bce(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.bce_logits = torch.nn.BCEWithLogitsLoss()

    def forward(self, dl, dlgt, sl, lgt):
        mask = lgt > 0

        _loss_seg = self.bce_logits(sl, lgt)
        _loss_seg[torch.isnan(_loss_seg)] = 0

        _loss_disp = F.smooth_l1_loss(dl[mask], dlgt.unsqueeze(1)[mask], reduction='mean')
        _loss_disp[torch.isnan(_loss_disp)] = 0

        _loss = _loss_disp + 2 * _loss_seg
        return _loss, _loss_disp, _loss_seg