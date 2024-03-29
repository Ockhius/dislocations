import torch
import torch.nn as nn
import torch.nn.functional as F
from delineation.models.blocks.blocks_3d import *
from delineation.models.blocks.blocks_2d import *
from delineation.layers.cost_volume import *


class SCMNET_SEG_JOINT(nn.Module):
    def __init__(self, in_channels, out_channels, num_block, maxdisp=16, disp_space='two-sided'):
        super(SCMNET_SEG_JOINT, self).__init__()
        self.disp_space = disp_space
        self.maxdisp = maxdisp
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.conv0 = nn.Conv2d(self.in_channels, 32, 5, 2, 2)
        self.in0 = nn.InstanceNorm2d(32)

        self.res_block = self._make_layer(res_2d_block_in, 32, 32, num_block[0], stride=1)

        self.conv1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.in1 = nn.InstanceNorm2d(32)

        # conv3d
        self.conv3d_1 = nn.Conv3d(64, 64, 3, 1, 1)
        self.bn3d_1 = nn.BatchNorm3d(64)
        self.conv3d_2 = nn.Conv3d(64, 64, 3, 1, 1)
        self.bn3d_2 = nn.BatchNorm3d(64)

        # conv3d sub_sample block
        self.block_3d_1 = self._make_layer(conv_3d_block, 64, 64, num_block[1], stride=2)
        self.block_3d_2 = self._make_layer(conv_3d_block, 64, 128, num_block[1], stride=2)

        # deconv3d
        self.deconv1 = nn.ConvTranspose3d(128, 64, 3, 2, 1, 1)
        self.debn1 = nn.BatchNorm3d(64)
        self.deconv2 = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.debn2 = nn.BatchNorm3d(64)

        # last disp deconv
        self.disp_deconv = nn.ConvTranspose3d(64, self.out_channels[0], 3, 2, 1, 1)

        # last seg deconv + conv
        self.seg_deconv = nn.ConvTranspose3d(64, 32, 3, 2, 1, 1)
        self.seg_conv = nn.Conv2d(32 * 2 * self.maxdisp, self.out_channels[1], 3, 1, 1)

    def forward(self, left_im, right_im):
        # N : Batchsize
        # D : 2 * maxdisp (two-sided)

        # feature extraction
        right_fmap = F.relu(self.in0(self.conv0(right_im)))
        left_fmap = F.relu(self.in0(self.conv0(left_im)))  # (N, 32, H/2, W/2)

        right_fmap = self.res_block(right_fmap)
        left_fmap = self.res_block(left_fmap)

        # feature reshape
        right_fmap = F.relu(self.in1(self.conv1(right_fmap)))
        left_fmap = F.relu(self.in1(self.conv1(left_fmap)))

        # cost volume
        cv = cost_volume(left_fmap, right_fmap,
                         maxdisp=self.maxdisp//2, disp_space=self.disp_space)  # (N, 64, D/2, H/2, W/2)

        conv3d_out = F.relu(self.bn3d_1(self.conv3d_1(cv)))
        conv3d_out = F.relu(self.bn3d_2(self.conv3d_2(conv3d_out)))  # (N, 32, D/2, H/2, W/2)

        conv3d_block_1 = self.block_3d_1(cv)  # (N, 64, D/4, H/4, W/4)
        conv3d_block_2 = self.block_3d_2(conv3d_block_1)  # (N, 32, D/8, H/8, W/8)

        # deconv
        deconv3d = F.relu(self.debn1(self.deconv1(conv3d_block_2)) + conv3d_block_1)  # (N, 64, D/4, H/4, W/4)
        deconv3d = F.relu(self.debn2(self.deconv2(deconv3d)) + conv3d_out)  # (N, 32, D/2, H/2, W/2)

        # disparity cost volume
        disp_cost_vol = self.disp_deconv(deconv3d)  # (N, 1, D, H, W)

        # segmentation
        deconv3d = self.seg_deconv(deconv3d)  # (N, 32, D, H, W)
        deconv3d = deconv3d.view(deconv3d.shape[0],
                                 deconv3d.shape[1] * deconv3d.shape[2],
                                 deconv3d.shape[3],
                                 deconv3d.shape[4])  # (N, 32 * D, H, W)
        segmap = self.seg_conv(deconv3d)

        return disp_cost_vol, segmap

    def _make_layer(self, block, in_planes, planes, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for step in strides:
            layers.append(block(in_planes, planes, step))

        return nn.Sequential(*layers)
