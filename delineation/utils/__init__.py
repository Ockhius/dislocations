import cv2
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn


def warp(image, disparity):
    ''' image : left_image (N, C, H, W)
        flow  : right disparity
        output : right image
        Requires CUDA available GPU'''

    # normalize flow-field vector
    H, W = image.size()[2], image.size()[3]
    disparity = disparity.permute(0,2,3,1)
    flow = torch.nn.ZeroPad2d((0,1,0,0))(disparity)
    flow = 2 * flow / W

    # create normalized meshgrid
    a = torch.linspace(-1.0, 1.0, H, dtype=torch.float32, requires_grad=False).cuda()
    b = torch.linspace(-1.0, 1.0, W, dtype=torch.float32, requires_grad=False).cuda()
    yy = a.view(-1,1).repeat(1, W)
    xx = b.repeat(H, 1)

    meshgrid = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2).unsqueeze_(0)

    # add flow to meshgrid
    pixloc = meshgrid + flow

    # sample grid
    warped_image = F.grid_sample(image, pixloc)
    return warped_image