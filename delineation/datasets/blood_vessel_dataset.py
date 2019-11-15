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


class BloodVesselDataset(Dataset):

    def __init__(self, rootdir, n_channels=3, transform=None, train=True, save_representations=False):

        self.root_dir = rootdir
        self.n_channels = n_channels
        self.transform = transform
        self.split = 'train' if train else 'test'
        self.save_representations = save_representations

        self.subset_folders = {'I':'images',
                               'L':'labels',
                               'M':'mask'}

        self.l_images_path = sorted(os.listdir(os.path.join(self.root_dir, self.split, self.subset_folders['I'])))

    def __len__(self):
        return len(self.l_images_path)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.root_dir, self.split, self.subset_folders['I'], self.l_images_path[idx]))
        label = Image.open(os.path.join(self.root_dir, self.split, self.subset_folders['L'], self.l_images_path[idx].replace('training.tif','manual1.png'))).convert('L')
        if self.n_channels == 1:
            image = image.convert('L')

        if self.split == 'train':

            r = random.randint(0, 100000)

            if (random.randint(0, 2) == 1):
                scaler = iaa.Affine(scale={"x": (0.7, 1.2)},   mode='reflect', random_state=r, deterministic=True)
                image = scaler.augment_image(np.array(image))
                label = scaler.augment_image(np.array(label))

            if (random.randint(0, 1) == 1):
                brightness = iaa.Multiply((0.8, 1.2))
                image = brightness.augment_image(np.array(image))

            if (random.randint(0, 2) == 1):
                rotate = iaa.Affine(rotate=(-60, 60),   mode='reflect', random_state=r, deterministic=True)
                image = rotate.augment_image(np.array(image))
                label = rotate.augment_image(np.array(label))

        image = np.array(image, dtype=np.float32)

        image = (image - image.mean())/(image.std()+1e-6)
        label = np.array(label, dtype=np.float32)

        if np.max(label) > 1:
            label = label / 255.0

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        if self.n_channels == 1:
            image = image.unsqueeze(0)
        else:
            image = image.permute(2,0,1)

        label = label.unsqueeze(0)

        return [image, label,  self.l_images_path[idx]]