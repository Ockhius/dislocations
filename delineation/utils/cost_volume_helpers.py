import argparse
import time
import random
import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use("TkAgg")
from mpl_toolkits.mplot3d import Axes3D


def volume_indices(D, B, H, W, device):
    indices = torch.arange(0, D, 1, dtype=torch.float32, requires_grad=False)
    indices = indices.unsqueeze(0).unsqueeze(0).unsqueeze(3).unsqueeze(3)
    indices = indices.expand(B, 1, D, H, W).to(device).contiguous()

    return indices


def back_project_numpy(lgt, dl, maxdisp=32, mode='two-sided'):
    ''' image : numpy ndarray (H, W)
        disparity : numpy ndarray (H, W, 1)
        output : numpy ndarray 3D volume (1, 2D, H, W) '''

    H, W = lgt.shape[0], lgt.shape[1]
    D = maxdisp
    dl = dl[:,:,0]
    print(dl)
    mask = (dl > -D-1) & (dl < D) *1
    print(mask)
    dl = np.clip(dl, -D, D-1)
    print(dl)
    dd = (dl + D).astype(int)
    yy = np.arange(0, H, 1)[:,np.newaxis].repeat(W, 1).astype(int)
    xx = np.arange(0, W, 1)[:,np.newaxis].transpose().repeat(H, 0).astype(int)

    if mode == 'two-sided':
        volume = np.zeros((1, 2*D, H, W))
    else:
        raise NotImplementedError
    volume[:, dd, yy, xx] = lgt * mask
    return volume


def visualize_volume(name, volume, rotation_angle, logdir, mode='scatter', save_ext=None, plot=False):

    if mode == 'scatter':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_ylim3d(0, volume.shape[2])
        ax.set_zlim3d(0, volume.shape[1])
        ax.view_init(0, rotation_angle)

        dd, yy, xx = np.nonzero(volume)

        ax.scatter(dd, xx, yy, c='r', marker='.')
        plt.gca().invert_zaxis()

        if save_ext is not None:
            save_path = logdir + '_results/' + name
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            plt.savefig(save_path + '/' + str(save_ext))
        if plot:
            plt.show()
            save_path = logdir+'results/'+name
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            plt.savefig(save_path+ '/'+str(rotation_angle))

        plt.close()