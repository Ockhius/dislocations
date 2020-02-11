import torch
import torch.nn.functional as F
from delineation.utils import *

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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


class smooth_l1_masked_dispartiy(torch.nn.Module):
    def __init__(self):

        super().__init__()

    def forward(self, dl, dlgt, lgt):

        mask = lgt > 0
        print('Max: {} Min: {}'.format(dlgt.unsqueeze(1)[mask].max(),
                                       dlgt.unsqueeze(1)[mask].min()))
        print(dl[mask].squeeze()[0:10])
        print(dlgt.unsqueeze(1)[mask].squeeze()[0:10])

        _loss = F.smooth_l1_loss(dl[mask], dlgt.unsqueeze(1)[mask], reduction='mean')
      #  _loss[torch.isnan(_loss)] = 0
        return _loss


class smooth_l1_disparity_and_edge_warp(torch.nn.Module):
    def __init__(self):

        super().__init__()

        self.bce_logits = torch.nn.BCEWithLogitsLoss()

    def forward(self, dl, seg_r, dlgt, lgt):

        mask = lgt > 0

        recon_l =  F.sigmoid(warp(seg_r, dl))
        recon_l_gt = F.sigmoid(warp(seg_r, dlgt.unsqueeze(1)))

        # recon_l_gt = recon_l_gt*mask
        recon_l = recon_l*mask

        # fig = plt.figure()
        #
        # plt.subplot(1, 4, 1)
        #
        # plt.imshow(recon_l[0].squeeze().data.cpu().numpy())
        #
        # plt.subplot(1, 4, 2)
        # plt.imshow(recon_l_gt[0].squeeze().data.cpu().numpy(), cmap='gray')
        #
        # plt.subplot(1, 4, 3)
        # plt.imshow(lgt[0].squeeze().data.cpu().numpy(), cmap='gray')
        #
        # plt.subplot(1, 4, 4)
        # plt.imshow(seg_r[0].squeeze().data.cpu().numpy(), cmap='gray')
        #
        # fig.set_size_inches(np.array(fig.get_size_inches()) * 3)
        #
        # plt.savefig('check_reconstruction.png'), \
        # plt.close()

        _loss_recon = F.smooth_l1_loss(recon_l[mask], lgt[mask], reduction='mean')

        _loss = F.smooth_l1_loss(dl[mask], dlgt.unsqueeze(1)[mask], reduction='mean')
        _loss[torch.isnan(_loss)] = 0

        return 0.5*_loss+0.5*_loss_recon


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