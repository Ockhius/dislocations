import torch
import torch.nn as nn
import torch.nn.functional as F
from delineation.models.blocks.blocks_3d import *
from delineation.models.blocks.blocks_2d import *
from delineation.layers.cost_volume import *


class ThreeDConv(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ThreeDConv, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        return out


class BasicBlock(nn.Module):  # basic block for Conv2d
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GC_NET(nn.Module):


    def __init__(self, maxdisp, in_channels, out_channels, disp_space, num_block=1, block_3d=ThreeDConv):
        super(GC_NET, self).__init__()
        self.disp_space = disp_space
        self.maxdisp = maxdisp
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.block = BasicBlock

        self.conv0 = nn.Conv2d(self.in_channels, 32, 5, 2, 2)
        self.bn0 = nn.BatchNorm2d(32)
        # res block
        #self.res_block = self._make_layer(self.block, self.in_planes, 32, 1, stride=1)
        # last conv2d
        #self.conv1 = nn.Conv2d(32, 32, 3, 1, 1)

        # conv3d
        self.conv3d_1 = nn.Conv3d(64, 32, 3, 1, 1)
        self.bn3d_1 = nn.BatchNorm3d(32)
        self.conv3d_2 = nn.Conv3d(32, 32, 3, 1, 1)
        self.bn3d_2 = nn.BatchNorm3d(32)

        self.conv3d_3 = nn.Conv3d(64, 64, 3, 2, 1)
        self.bn3d_3 = nn.BatchNorm3d(64)
        self.conv3d_4 = nn.Conv3d(64, 64, 3, 2, 1)
        self.bn3d_4 = nn.BatchNorm3d(64)
        self.conv3d_5 = nn.Conv3d(64, 64, 3, 2, 1)
        self.bn3d_5 = nn.BatchNorm3d(64)

        # conv3d sub_sample block
        self.block_3d_1 = self._make_layer(block_3d, 64, 64, num_block, stride=2)
        self.block_3d_2 = self._make_layer(block_3d, 64, 64, num_block, stride=2)
        self.block_3d_3 = self._make_layer(block_3d, 64, 64, num_block, stride=2)
        self.block_3d_4 = self._make_layer(block_3d, 64, 128, num_block, stride=2)
        # deconv3d
        self.deconv1 = nn.ConvTranspose3d(128, 64, 3, 2, 1, 1)
        self.debn1 = nn.BatchNorm3d(64)
        self.deconv2 = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.debn2 = nn.BatchNorm3d(64)
        self.deconv3 = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.debn3 = nn.BatchNorm3d(64)
        self.deconv4 = nn.ConvTranspose3d(64, 32, 3, 2, 1, 1)
        self.debn4 = nn.BatchNorm3d(32)
        # last deconv3d
        self.deconv5 = nn.ConvTranspose3d(32, self.out_channels, 3, 2, 1, 1)

    def cost_volume(self, ref, target, disp_space='two-sided', maxdisp=16):
        if disp_space == 'two-sided':
            lowest = - maxdisp
            highest = maxdisp
            disp_size = 2 * maxdisp
        else:
            lowest = 0
            highest = maxdisp
            disp_size = maxdisp

        # N x 2F x D/2 x H/2 x W/2
        cost = torch.FloatTensor(ref.size()[0],
                                 ref.size()[1] * 2,
                                 disp_size,
                                 ref.size()[2],
                                 ref.size()[3]).zero_().cuda()

        # reference is the left image, target is the right image. IL (+) DL = IR
        for i, k in enumerate(range(lowest, highest)):
            if k == 0:
                cost[:, :ref.size()[1], i, :, :] = ref
                cost[:, ref.size()[1]:, i, :, :] = target
            elif k < 0:
                cost[:, :ref.size()[1], i, :, -k:] = ref[:, :, :, -k:]
                cost[:, ref.size()[1]:, i, :, -k:] = target[:, :, :, :k]  # target shifted to right over ref
            else:
                cost[:, :ref.size()[1], i, :, :-k] = ref[:, :, :, :-k]
                cost[:, ref.size()[1]:, i, :, :-k] = target[:, :, :, k:]  # target shifted to left over ref

        return cost.contiguous()

    def forward(self, seg_rep_left, seg_rep_right):

        right_fmap = F.relu(self.bn0(self.conv0(seg_rep_right)))
        left_fmap = F.relu(self.bn0(self.conv0(seg_rep_left)))

        # cost volume

        cvolume = self.cost_volume(left_fmap, right_fmap, self.disp_space, self.maxdisp //2)

        conv3d_out = F.relu(self.bn3d_1(self.conv3d_1(cvolume)))
        conv3d_out = F.relu(self.bn3d_2(self.conv3d_2(conv3d_out)))

        conv3d_block_1 = self.block_3d_1(cvolume)
        conv3d_21 = F.relu(self.bn3d_3(self.conv3d_3(cvolume)))
        conv3d_block_2 = self.block_3d_2(conv3d_21)
        conv3d_24 = F.relu(self.bn3d_4(self.conv3d_4(conv3d_21)))
        conv3d_block_3 = self.block_3d_3(conv3d_24)
        conv3d_27 = F.relu(self.bn3d_5(self.conv3d_5(conv3d_24)))
        conv3d_block_4 = self.block_3d_4(conv3d_27)

        # deconv
        deconv3d = F.relu(self.debn1(self.deconv1(conv3d_block_4)) + conv3d_block_3)
        deconv3d = F.relu(self.debn2(self.deconv2(deconv3d)) + conv3d_block_2)
        deconv3d = F.relu(self.debn3(self.deconv3(deconv3d)) + conv3d_block_1)
        deconv3d = F.relu(self.debn4(self.deconv4(deconv3d)) + conv3d_out)

        deconv3d = self.deconv5(deconv3d)
        return deconv3d

    def _make_layer(self, block, in_planes, planes, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for step in strides:
            layers.append(block(in_planes, planes, step))

        return nn.Sequential(*layers)



class SCMNET(nn.Module):
    def __init__(self, in_channels, out_channels, num_block, maxdisp=32, disp_space='two-sided'):
        super(SCMNET, self).__init__()
        self.disp_space = disp_space
        self.maxdisp = maxdisp
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.conv0 = nn.Conv2d(self.in_channels, 32, 5, 2, 2)
        self.bn0 = nn.BatchNorm2d(32)

        self.res_block = self._make_layer(res_2d_block, 32, 32, num_block[0], stride=1)

        self.conv1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)

        # conv3d
        self.conv3d_1 = nn.Conv3d(64, 32, 3, 1, 1)
        self.bn3d_1 = nn.BatchNorm3d(32)
        self.conv3d_2 = nn.Conv3d(32, 32, 3, 1, 1)
        self.bn3d_2 = nn.BatchNorm3d(32)

        # conv3d sub_sample block
        self.block_3d_1 = self._make_layer(conv_3d_block, 64, 64, num_block[1], stride=2)
        self.block_3d_2 = self._make_layer(conv_3d_block, 64, 64, num_block[1], stride=2)

        # deconv3d
        self.deconv1 = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.debn1 = nn.BatchNorm3d(64)
        self.deconv2 = nn.ConvTranspose3d(64, 32, 3, 2, 1, 1)
        self.debn2 = nn.BatchNorm3d(32)
        # last deconv3d
        self.deconv5 = nn.ConvTranspose3d(32, self.out_channels, 3, 2, 1, 1)

    def forward(self, left_im, right_im):
        # N : Batchsize
        # D : 2 * maxdisp (two-sided)

        # feature extraction
        right_fmap = F.relu(self.bn0(self.conv0(right_im)))
        left_fmap = F.relu(self.bn0(self.conv0(left_im)))  # (N, 32, H/2, W/2)

        # right_fmap = self.res_block(right_fmap)
        # left_fmap = self.res_block(left_fmap)
        #
        # # feature reshape
        # right_fmap = F.relu(self.bn1(self.conv1(right_fmap)))
        # left_fmap = F.relu(self.bn1(self.conv1(left_fmap)))

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
        deconv3d = self.deconv5(deconv3d)  # (N, 1, D, H, W)

        return deconv3d

    def _make_layer(self, block, in_planes, planes, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for step in strides:
            layers.append(block(in_planes, planes, step))

        return nn.Sequential(*layers)
