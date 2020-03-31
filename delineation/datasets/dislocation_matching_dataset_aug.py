from __future__ import print_function, division
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import random

from delineation.configs.defaults_segmentation import _C as cfg
from delineation.datasets import data_augmentation as data_aug
import cv2

class MatchingDislocationsDataset(Dataset):

    def __init__(self, cfg, cfg_aug, transform=None, train=True):

        self.root_dir = cfg.INPUT.SOURCE
        self.transform = transform
        self.split = 'train' if train else 'test'
        #cfg file for augmentations
        self.cfg_aug = cfg_aug

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

    def apply_augmentations(self, image, augmentations, seed=None):
        def apply_augmentation(image, aug, augmentations, seed=None):
            '''
            arguments: image - gray image with intensity scale 0-255
                        aug - name of the augmentation to apply
                        seed
            output:
                        image - gray image after augmentation with intensity scale 0-255
            '''
            if aug == 'additive_gaussian_noise':
                image, kp = data_aug.additive_gaussian_noise(image, [], seed=seed,
                                                    std=(augmentations['ADDITIVE_GAUSSIAN_NOISE']['STD_MIN'],
                                                         augmentations['ADDITIVE_GAUSSIAN_NOISE']['STD_MAX']))
            if aug == 'additive_speckle_noise':
                image, kp = data_aug.additive_speckle_noise(image, [], intensity=augmentations['ADDITIVE_SPECKLE_NOISE']['INTENSITY'])
            if aug == 'random_brightness':
                image, kp = data_aug.random_brightness(image, [], seed=seed)
            if aug == 'random_contrast':
                image, kp = data_aug.random_contrast(image, [], seed=seed)
            if aug == 'add_shade':
                image, kp = data_aug.add_shade(image, [], seed=seed)
            if aug == 'motion_blur':
                image, kp = data_aug.motion_blur(image, [], max_ksize=augmentations['MOTION_BLUR']['MAX_KSIZE'])
            if aug == 'gamma_correction':
                # must be applied on image with intensity scale 0-1
                maximum = np.max(image)
                if maximum != 0:
                    image_preprocessed = image / maximum if maximum > 0 else 0
                    random_gamma = random.uniform(augmentations['GAMMA_CORRECTION']['MIN_GAMMA'], \
                                                  augmentations['GAMMA_CORRECTION']['MAX_GAMMA'])
                    image_preprocessed = image_preprocessed ** random_gamma
                    image = image_preprocessed * maximum
            if aug == 'opposite':
                # must be applied on image with intensity scale 0-1
                maximum = np.max(image)
                if maximum != 0:
                    image_preprocessed = image / maximum if maximum > 0 else 0
                    image_preprocessed = 1 - image_preprocessed
                    image = image_preprocessed * maximum
            if aug == 'no_aug':
                pass
            return image

        random.seed(seed)
        list_of_augmentations = augmentations['AUGMENTATION_LIST']
        index = random.sample(range(len(list_of_augmentations)), 3)
        for i in index:
            aug = list_of_augmentations[i]
            image = apply_augmentation(image, aug, augmentations, seed)

        image_preprocessed = image / (np.max(image) + 0.000001)
        return image, image_preprocessed

    def __getitem__(self, idx):

        seed = np.random.randint(0,100000)
        random.seed(seed)

        left_img = Image.open(os.path.join(self.root_dir, self.split, self.subset_folders['L'], self.l_images_path[idx])).convert('L')
        right_img = Image.open(os.path.join(self.root_dir, self.split, self.subset_folders['R'], self.l_images_path[idx].replace('LEFT','RIGHT'))).convert('L')
        left_gt_img = Image.open(os.path.join(self.root_dir, self.split, self.subset_folders['S_L'], self.l_images_path[idx])).convert('L')
        right_gt_img = Image.open(os.path.join(self.root_dir, self.split, self.subset_folders['S_R'], self.l_images_path[idx].replace('LEFT','RIGHT'))).convert('L')

        left_img, image1_preprocessed = self.apply_augmentations(np.array(left_img, dtype=np.float32), self.cfg_aug['TRAINING']['AUGMENTATION'], seed=seed * (idx + 1))
        right_img, image1_preprocessed = self.apply_augmentations(np.array(right_img, dtype=np.float32), self.cfg_aug['TRAINING']['AUGMENTATION'], seed=seed * (idx + 1))

        left_disp_gt = np.array(Image.open(os.path.join(self.root_dir, self.split, self.subset_folders['D'], self.l_images_path[idx])).convert('L'), dtype=np.float32)-127

        left_gt_img = np.array(left_gt_img, dtype=np.float32)
        right_gt_img = np.array(right_gt_img, dtype=np.float32)

        left_gt_img[left_gt_img < 128] = 0
        right_gt_img[right_gt_img < 128] = 0
        left_disp_gt[left_disp_gt==-127]=0

        # if self.cfg_aug['TRAINING']['TRANSLATION_AUG'] == True:
        #
        #     translation = np.random.randint(-15,15)
        #     num_rows, num_cols = np.array(left_img).shape
        #
        #     T = np.float32([[1, 0, translation], [0, 1, 0]])
        #     right_img = cv2.warpAffine(np.array(right_img), T, (num_cols, num_rows))
        #     right_gt_img = cv2.warpAffine(np.array(right_gt_img), T, (num_cols, num_rows))
        #     left_disp_gt=left_disp_gt+translation

        left_img = np.array(left_img, dtype=np.float32) / 255.0
        right_img = np.array(right_img, dtype=np.float32) / 255.0

        if np.max(left_gt_img) > 1:
            left_gt_img = left_gt_img / 255.0

        if np.max(right_gt_img) > 1:
            right_gt_img = right_gt_img / 255.0


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