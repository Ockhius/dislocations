import tifffile
import os
from PIL import Image
import numpy as np
# create folders to save converted images

def create_folders(save_path, save_folders):

    for folder in save_folders:
        save_path_ = os.path.join(save_path, folder)
        if not os.path.exists(save_path_):
            os.makedirs(save_path_)

def create_spade_dataset(retina_images_folder, save_path, save_folders):

    for img_name in os.listdir(retina_images_folder):

        if 'label' not in img_name:
            label = tifffile.imread(os.path.join(retina_images_folder, img_name.replace('.tiff', 'label.tiff')))[:, :, 0:3]
            img = tifffile.imread(os.path.join(retina_images_folder, img_name))[:, :, 0:3]

            Image.fromarray(img).save(os.path.join(save_path, save_folders[0], img_name.replace('tiff','png')))
            label[np.nonzero(label)]=255
            Image.fromarray(label).save(os.path.join(save_path, save_folders[1], img_name.replace('tiff','png')))
            print('')
if __name__ == '__main__':

    retina_images_folder = '/cvlabsrc1/cvlab/datasets_anastasiia/retina/deepdyn/data/DRIVE_cropped'

    save_path = '/cvlabsrc1/cvlab/datasets_anastasiia/Dislocations/SPADE/datasets/retina/'
    save_folders = ['images', 'labels']

    create_folders(save_path, save_folders)
    create_spade_dataset(retina_images_folder, save_path, save_folders)