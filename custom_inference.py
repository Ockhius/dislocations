config_file = 'delineation/configs/dislocation_matching_disp_and_warp_and var_joint.yml'
aug_config_file = 'delineation/configs/aug.yml'
import torch
from delineation.configs.defaults_segmentation import _C as cfg
from delineation.datasets import make_data_loader
from delineation.models import build_model_list
from delineation.utils import settings, cost_volume_helpers
from delineation.utils.settings import evaluate_results
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import os
import yaml

cfg.merge_from_file(config_file)
with open(aug_config_file, 'r') as ymlfile:
    cfg_aug = yaml.load(aug_config_file)

_device = settings.initialize_cuda_and_logging(cfg)  # '_device' is GLOBAL VAR

train_loader, val_loader = make_data_loader(cfg, cfg_aug)
seg_model, model = build_model_list(cfg, True)
# %%
seg_model.eval()
model.eval()

target = '3_45im_-48+50_pair_0_1_10018_10024.png'

l = cv2.imread('/cvlabsrc1/cvlab/datasets_anastasiia/dislocations/ALL_DATA_fixed_bottom_img_with_semantics_resized_1024/train/left_image/3_45im_-48+50_pair_18_19_10018_LEFT.png', 0)
r = cv2.imread('/cvlabsrc1/cvlab/datasets_anastasiia/dislocations/ALL_DATA_fixed_bottom_img_with_semantics_resized_1024/train/left_image/3_45im_-48+50_pair_24_25_10024_LEFT.png', 0)

        # l = cv2.resize(l, (512, 512))
        # r = cv2.resize(r, (512, 512))

l = torch.from_numpy(l).unsqueeze(0).unsqueeze(0) / 255.0
r = torch.from_numpy(r).unsqueeze(0).unsqueeze(0) / 255.0

with torch.no_grad():
            device = torch.device('cuda')
            seg_model = seg_model.to(device)

            indices = cost_volume_helpers.volume_indices(2 * cfg.TRAINING.MAXDISP, len(l),
                                                         1024, 1024, _device)

            l_segmap, l_seg = seg_model(l.cuda())
            r_segmap, r_seg = seg_model(r.cuda())

            dl_scores = model(l_segmap.cuda(), r_segmap.cuda())
            l_seg = l_seg.cpu().numpy() > 0.1

            dl_ = F.softmax(-dl_scores, 2)
            dl = torch.sum(dl_.mul(indices), 2) - cfg.TRAINING.MAXDISP

            dl = dl.detach().permute(0, 2, 3, 1).cpu().numpy()

            for i in tqdm(range(len(dl))):
                vol = cost_volume_helpers.back_project_numpy(l_seg[i, 0, :, :], dl[i, :, :, :], cfg.TRAINING.MAXDISP,
                                                             mode='two-sided')

                save_path = cfg.LOGGING.LOG_DIR + '_results/' + target

                np.save(save_path+target+'.npy', vol)
                for idx, rotation_angle in enumerate(range(-20, 21, 1)):
                    cost_volume_helpers.visualize_volume(target, vol[0, :, :, :],
                                                         rotation_angle,
                                                         cfg.LOGGING.LOG_DIR,
                                                         mode='scatter',
                                                         save_ext=str(idx) + '.png',
                                                         plot=False)

        # %%
