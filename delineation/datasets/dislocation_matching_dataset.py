from __future__ import print_function, division
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from delineation.configs.defaults_segmentation import _C as cfg


class MatchingDislocationsDataset(Dataset):

    def __init__(self, cfg, transform=None, train=True):

        self.root_dir = cfg.INPUT.SOURCE
        self.transform = transform
        self.split = 'train' if train else 'test'

        self.subset_folders = {'L':'left_image',
                               'R':'right_image',
                               'S_R':'segmentation_right_image',
                               'S_L':'segmentation_left_image',
                               'S_L_P':'left_predictions',
                               'S_R_P':'right_predictions',
                               'D':'disparity',
                               'L_R':'left_representations',
                               'R_R':'right_representations'
                               }

        self.l_images_path = sorted(os.listdir(os.path.join(self.root_dir, self.split, self.subset_folders['L'])))

    def __len__(self):
        return len(self.l_images_path)

    def normalize_image(self, img):

        img = np.array(img, dtype=np.float32)
        if np.max(img) > 1:
            img = img / 255.0

        return img

    def __getitem__(self, idx):

        left_img = Image.open(os.path.join(self.root_dir, self.split, self.subset_folders['L'], self.l_images_path[idx])).convert('L')
        right_img = Image.open(os.path.join(self.root_dir, self.split, self.subset_folders['R'], self.l_images_path[idx].replace('LEFT','RIGHT'))).convert('L')
        left_gt_img = Image.open(os.path.join(self.root_dir, self.split, self.subset_folders['S_L'], self.l_images_path[idx])).convert('L')
        right_gt_img = Image.open(os.path.join(self.root_dir, self.split, self.subset_folders['S_R'], self.l_images_path[idx].replace('LEFT','RIGHT'))).convert('L')

        left_disp_gt = np.array(Image.open(os.path.join(self.root_dir, self.split, self.subset_folders['D'], self.l_images_path[idx])).convert('L'), dtype=np.float32)-127

        left_img = np.array(left_img, dtype=np.float32) / 255.0
        right_img = np.array(right_img, dtype=np.float32) / 255.0

        left_gt_img = np.array(left_gt_img, dtype=np.float32)
        right_gt_img = np.array(right_gt_img, dtype=np.float32)

        left_gt_img[left_gt_img < 128] = 0
        right_gt_img[right_gt_img < 128] = 0

        if np.max(left_gt_img) > 1:
            left_gt_img = left_gt_img / 255.0

        if np.max(right_gt_img) > 1:
            right_gt_img = right_gt_img / 255.0

        left_disp_gt[left_disp_gt==-127]=0

        left_img = torch.from_numpy(left_img).float().unsqueeze(0)
        right_img = torch.from_numpy(right_img).float().unsqueeze(0)
        left_seg_img = torch.from_numpy(left_gt_img).float().unsqueeze(0)
        right_seg_img = torch.from_numpy(right_gt_img).float().unsqueeze(0)

        return [left_img, right_img, left_seg_img, right_seg_img, left_disp_gt, self.l_images_path[idx]]


if __name__ == '__main__':

        cfg.merge_from_file('C:\\Users\\okana\\workspace\\projects\\dislocations\\delineation\\configs\\dislocation_segmentation_home.yml')
        dataset = MatchingDislocationsDataset(cfg, train=True)
        l, r, lgt, rgt, dlgt, _ = dataset[19]

        ax = plt.subplot(3, 2, 1)
        im = ax.imshow(l.data.numpy()[0], cmap='gray')

        ax = plt.subplot(3, 2, 2)
        im = ax.imshow(r.data.numpy()[0], cmap='gray')

        ax = plt.subplot(3, 2, 3)
        im = ax.imshow(lgt.data.numpy()[0], cmap='gray')

        ax = plt.subplot(3, 2, 4)
        im = ax.imshow(rgt.data.numpy()[0], cmap='gray')

        print(np.mean(np.abs(dlgt[dlgt.nonzero()])))
        print(np.max(dlgt))
        print(np.min(dlgt))
        ax = plt.subplot(3, 2, 5)
        im = ax.imshow(dlgt, cmap='gray')

        plt.show()