import torch
from delineation.utils import *

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from torch import nn

def bce_segmentation_loss(projection, gt):
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(projection, gt)
    return loss

def initia_loss_func(seg_l, lgt, seg_r, rgt):

    loss = 0.5 * bce_segmentation_loss(seg_l, lgt) \
           + 0.5 * bce_segmentation_loss(seg_r, rgt)

    return loss


class binary_cross_entropy(torch.nn.Module):
    def __init__(self, ignore_index=250):

        super().__init__()

        self.left_bce_logits = torch.nn.BCEWithLogitsLoss()
        self.right_bce_logits = torch.nn.BCEWithLogitsLoss()

    def forward(self, seg_l, lgt, seg_r, rgt):
        _loss = 0.5 * self.left_bce_logits(seg_l, lgt) + 0.5 * self.right_bce_logits(seg_r, rgt)
        return _loss

class smooth_l1_masked_disparity(torch.nn.Module):
    def __init__(self):

        super().__init__()

    def forward(self, dl, seg_r, dlgt, lgt):

        mask = lgt > 0
        print('Max: {} Min: {}'.format(dlgt.unsqueeze(1)[mask].max(),
                                       dlgt.unsqueeze(1)[mask].min()))

        print('Predicted Max: {} Min: {}'.format(dl[mask].max(),
                                       dl[mask].min()))

        print(dl[mask].squeeze()[0:10])
        print(dlgt.unsqueeze(1)[mask].squeeze()[0:10])

        _loss = F.smooth_l1_loss(dl[mask], dlgt.unsqueeze(1)[mask], reduction='mean')

        return _loss

class warp_only(torch.nn.Module):
    def __init__(self):

        super().__init__()

        self.bce_logits = torch.nn.BCEWithLogitsLoss()

    def forward(self, dl, seg_r, dlgt, lgt):

        mask = lgt > 0
        recon_l =  F.sigmoid(warp(seg_r, dl))

        recon_l = recon_l*mask

        _loss_recon = F.smooth_l1_loss(recon_l[mask], lgt[mask], reduction='mean')

        return _loss_recon

class warp_only_joint(torch.nn.Module):
    def __init__(self):

        super().__init__()

        self.bce_logits = torch.nn.BCEWithLogitsLoss()

    def forward(self, l_aug, r_aug, l_gt_aug, r_gt_aug, dl, seg_r, seg_l, dlgt, lgt, rgt):

        loss_seg1 = 0.5 * bce_segmentation_loss(l_aug, l_gt_aug) \
               + 0.5 * bce_segmentation_loss(r_aug, l_gt_aug)

        recon_l =  warp(seg_r, dl)*seg_l
        _loss_recon = F.smooth_l1_loss(recon_l, lgt, reduction='mean')

        return 0.5*loss_seg1 + _loss_recon

class smooth_l1_masked_disparity_joint(torch.nn.Module):
    def __init__(self):

        super().__init__()

    def forward(self, l_aug, r_aug, l_gt_aug, r_gt_aug, dl, seg_r, seg_l, dlgt, lgt, rgt):

        loss_seg1 = 0.5 * bce_segmentation_loss(l_aug, l_gt_aug) \
               + 0.5 * bce_segmentation_loss(r_aug, l_gt_aug)

        mask = lgt > 0
        print('Max: {} Min: {}'.format(dlgt.unsqueeze(1)[mask].max(),
                                       dlgt.unsqueeze(1)[mask].min()))

        print('Predicted Max: {} Min: {}'.format(dl[mask].max(),
                                       dl[mask].min()))

        print(dl[mask].squeeze()[0:10])
        print(dlgt.unsqueeze(1)[mask].squeeze()[0:10])

        _loss = F.smooth_l1_loss(dl[mask], dlgt.unsqueeze(1)[mask], reduction='mean')
        _loss[torch.isnan(_loss)] = 0

        return 0.5*loss_seg1+ _loss


class smooth_l1_disparity_and_edge_warp_joint(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.bce_logits = torch.nn.BCEWithLogitsLoss()

    def forward(self, l_aug, r_aug, l_gt_aug, r_gt_aug, dl, seg_r, seg_l, dlgt, lgt, rgt):

        loss_seg1 = 0.5 * bce_segmentation_loss(l_aug, l_gt_aug) \
               + 0.5 * bce_segmentation_loss(r_aug, l_gt_aug)

        mask = lgt > 0

        recon_l =  warp(F.sigmoid(seg_r), dl)*seg_l

        _loss_recon = F.smooth_l1_loss(recon_l, lgt, reduction='mean')

        _loss = F.smooth_l1_loss(dl[mask], dlgt.unsqueeze(1)[mask], reduction='mean')
        _loss[torch.isnan(_loss)] = 0

        return 0.5*loss_seg1+_loss+ 0.5*_loss_recon

class smooth_l1_disparity_and_edge_warp(torch.nn.Module):
    def __init__(self):

        super().__init__()

        self.bce_logits = torch.nn.BCEWithLogitsLoss()

    def forward(self, dl, seg_r, dlgt, lgt):

        mask = lgt > 0
        recon_l =  F.sigmoid(warp(seg_r, dl))
     #   recon_l_gt = F.sigmoid(warp(seg_r, dlgt.unsqueeze(1)))

        # recon_l_gt = recon_l_gt*mask
        recon_l = recon_l*mask
        _loss_recon = F.smooth_l1_loss(recon_l[mask], lgt[mask], reduction='mean')

        _loss = F.smooth_l1_loss(dl[mask], dlgt.unsqueeze(1)[mask], reduction='mean')
        _loss[torch.isnan(_loss)] = 0

        return _loss+ 0.5*_loss_recon


class smooth_l1_disparity(torch.nn.Module):
    def __init__(self):

        super().__init__()

        self.bce_logits = torch.nn.BCEWithLogitsLoss()

    def forward(self, dl, seg_r, dlgt, lgt):

        mask = lgt > 0
        recon_l =  F.sigmoid(warp(seg_r, dl))

        # recon_l_gt = recon_l_gt*mask
        recon_l = recon_l*mask

        _loss_recon = F.smooth_l1_loss(recon_l[mask], lgt[mask], reduction='mean')

        _loss = F.smooth_l1_loss(dl[mask], dlgt.unsqueeze(1)[mask], reduction='mean')
        _loss[torch.isnan(_loss)] = 0

        return _loss


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