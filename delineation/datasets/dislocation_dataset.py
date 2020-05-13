from __future__ import print_function, division
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from delineation.configs.defaults_segmentation import _C as cfg
from imgaug import augmenters as iaa
import random
import cv2

cv2.setNumThreads(0)

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose,RandomShadow
)

import numpy as np

def strong_aug(p=0.5):
    return Compose([
        RandomShadow(p=0.1),
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.01),
        OneOf([
            MotionBlur(p=0.4),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.4, rotate_limit=45, p=0.4),
        OneOf([
            RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),
        ], p=0.3),
    ], p=p)


class DislocationDataset(Dataset):

    def __init__(self, cfg, transform=None, train = True, save_representations=False):

        self.cfg = cfg
        self.root_dir = cfg.INPUT.SOURCE
        self.transform = transform
        self.split = 'train' if train else 'val'
        self.save_representations = save_representations

        self.subset_folders = {'L':'left_image',
                               'R':'right_image',
                               'S_R':'segmentation_right_image',
                               'S_L':'segmentation_left_image'}

        self.l_images_path = sorted(os.listdir(os.path.join(self.root_dir, self.split, self.subset_folders['L'])))

    def __len__(self):
        return len(self.l_images_path)

    def __getitem__(self, idx):

        left_img = Image.open(os.path.join(self.root_dir, self.split, self.subset_folders['L'], self.l_images_path[idx])).convert('L')
        right_img = Image.open(os.path.join(self.root_dir, self.split, self.subset_folders['R'], self.l_images_path[idx].replace('LEFT','RIGHT'))).convert('L')

        left_gt_img = Image.open(os.path.join(self.root_dir, self.split, self.subset_folders['S_L'], self.l_images_path[idx]))
        right_gt_img = Image.open(os.path.join(self.root_dir, self.split, self.subset_folders['S_R'], self.l_images_path[idx].replace('LEFT','RIGHT')))

        if self.cfg.TRAINING.NUM_CHANNELS == 3:
            left_img, right_img = left_img.convert('RGB'), right_img.convert('RGB')

        if self.split == 'train' and not self.save_representations:

            left_img, right_img = left_img.convert('RGB'), right_img.convert('RGB')

            augmentation = strong_aug(p=0.9)
            data_l = {"image": np.array(left_img), "mask": np.array(left_gt_img)}
            data_r = {"image": np.array(right_img), "mask": np.array(right_gt_img)}

            augmented_l = augmentation(**data_l)
            augmented_r = augmentation(**data_r)

            left_img,  left_gt_img= augmented_l["image"], augmented_l["mask"]
            right_img,  right_gt_img= augmented_r["image"], augmented_r["mask"]

            if self.cfg.TRAINING.NUM_CHANNELS == 1:
                left_img, right_img = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY), cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)

        left_img = np.array(left_img, dtype=np.float32) / 255.0
        right_img = np.array(right_img, dtype=np.float32) / 255.0

        # left_img = (left_img  - left_img.mean())/(left_img.std()+1e-6)
        # right_img = (right_img  - right_img.mean())/(right_img.std()+1e-6)

        left_gt_img = np.array(left_gt_img, dtype=np.float32)
        right_gt_img = np.array(right_gt_img, dtype=np.float32)

        left_gt_img[left_gt_img < 128] = 0
        right_gt_img[right_gt_img < 128] = 0

        if np.max(left_gt_img) > 1:
            left_gt_img = left_gt_img / 255.0

        if np.max(right_gt_img) > 1:
            right_gt_img = right_gt_img / 255.0

        left_img = torch.from_numpy(left_img).float()
        right_img = torch.from_numpy(right_img).float()
        left_gt_img = torch.from_numpy(left_gt_img).float()
        right_gt_img = torch.from_numpy(right_gt_img).float()

        if self.cfg.TRAINING.NUM_CHANNELS == 1:

            left_img = left_img.unsqueeze(0)
            right_img = right_img.unsqueeze(0)

        else:
            left_img = left_img.permute(2,0,1)
            right_img = right_img.permute(2,0,1)

        left_gt_img = left_gt_img.unsqueeze(0)
        right_gt_img = right_gt_img.unsqueeze(0)

        return [left_img, right_img, left_gt_img, right_gt_img,  self.l_images_path[idx]]


if __name__ == '__main__':

        #cfg.merge_from_file('C:\\Users\\okana\\workspace\\projects\\dislocations\\delineation\\configs\\dislocation_segmentation_home.yml')
        dataset = DislocationDataset(rootdir="D:/Datasets/dislocations/ALL_DATA_fixed_bottom_resized/", train=True)
        l, r, lgt, rgt, _ = dataset[0]

        ax = plt.subplot(2, 2, 1)
        im = ax.imshow(l.data.numpy()[0], cmap='gray')

        ax = plt.subplot(2, 2, 2)
        im = ax.imshow(r.data.numpy()[0], cmap='gray')

        ax = plt.subplot(2, 2, 3)
        im = ax.imshow(lgt.data.numpy()[0], cmap='gray')

        ax = plt.subplot(2, 2, 4)
        im = ax.imshow(rgt.data.numpy()[0], cmap='gray')

        plt.show()