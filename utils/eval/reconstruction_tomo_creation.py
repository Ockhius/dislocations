import numpy as np
import torch
from tqdm import tqdm
from delineation.utils import cost_volume_helpers
from tifffile import imsave

angle = 2
vol_size = 309
save_name = 'stack_reconstructed_features_618_0_21_22.tif'

dl_image = np.load(
    '3_45im_-48+50_pair_21_22_Aligned 45 of 450022_LEFT.png_dl_.npy')
seg_map_ = np.load(
    '3_45im_-48+50_pair_21_22_Aligned 45 of 450022_LEFT.png_segl_.npy')


def depth_from_disp(angle_gap, disp_im):
    angle_rad = np.pi * angle_gap / 180
    H, W = disp_im.shape
    x = np.arange(-W // 2, W // 2)[np.newaxis, :].repeat(H, axis=0)
    depth_im = (disp_im - x * np.cos(angle_rad) + x) / np.sin(angle_rad)
    return depth_im

depth_map = depth_from_disp(angle, dl_image.squeeze())

dl_ = torch.from_numpy(depth_map).unsqueeze(0).unsqueeze(0).permute(0, 2, 3, 1).cpu().numpy()
max_disp = int(max(abs(dl_.min()), dl_.max()))
print(max_disp)

vol = cost_volume_helpers.back_project_numpy(seg_map_[0, 0, :, :], dl_[0, :, :, :], vol_size, mode='two-sided')[0]

print(dl_.min(), dl_.max())

stack_of_images = []

for i in tqdm(range(len(vol) - 1, -1, -1)):
    img_i = np.array(vol[i]) * 255.0
    stack_of_images.append(np.array(img_i, dtype=np.uint8))

imsave(save_name, np.array(stack_of_images, dtype=np.uint8))