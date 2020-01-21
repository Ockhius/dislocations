import random
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def evaluate_results(r_gt_disp, r_pred_disp, r_segm):

    r_gt_disp = r_gt_disp.squeeze().data.cpu().numpy()
    r_pred_disp = r_pred_disp.squeeze().data.cpu().numpy()
    r_segm = r_segm.squeeze().data.cpu().numpy()

    mask = r_segm == 1

    pix1_err = np.sum((np.abs(r_pred_disp*mask - r_gt_disp*mask) > 1)) / np.sum(mask)
    pix3_err = np.sum((np.abs(r_pred_disp*mask - r_gt_disp*mask) > 3)) / np.sum(mask)
    pix5_err = np.sum((np.abs(r_pred_disp*mask - r_gt_disp*mask) > 5)) / np.sum(mask)

    epe = np.mean(np.abs(r_pred_disp[mask] - r_gt_disp[mask]))

    return pix1_err, pix3_err, pix5_err, epe

def debug_segmentation_val(left_img, leftpred, leftgt, filename=None):
    left_img, leftpred, leftgt = left_img.squeeze(0), leftpred.squeeze(0), leftgt.squeeze(0)

    fig = plt.figure()

    plt.subplot(1, 3, 1)
    if left_img.shape[0] > 1 and len(left_img.shape)>2:
        plt.imshow(left_img.data.cpu().numpy().transpose(1, 2, 0))
    else:
        plt.imshow(left_img.data.cpu().numpy(), cmap='gray')

    plt.subplot(1, 3, 2)
    plt.imshow(leftgt.data.cpu().numpy(), cmap='gray')

    plt.subplot(1, 3, 3)
    plt.imshow(leftpred.data.cpu().numpy(), cmap='gray')

    fig.set_size_inches(np.array(fig.get_size_inches()) * 3)

    plt.savefig(filename), plt.close()


def initialize_cuda_and_logging (cfg):

    print(("NOT " if not cfg.TRAINING.CUDA else "") + "Using cuda")
    # add experiment name to the folder names
    LOG_DIR = os.path.join(cfg.LOGGING.LOG_DIR, cfg.TRAINING.EXPERIMENT_NAME)
    MODELS_DIR = os.path.join(cfg.TRAINING.MODEL_DIR, cfg.TRAINING.EXPERIMENT_NAME)

    cfg.LOGGING.LOG_DIR = LOG_DIR
    cfg.TRAINING.MODEL_DIR = MODELS_DIR

    if cfg.TRAINING.CUDA:
        if not torch.cuda.is_available():
            raise Exception("CUDA is NOT available!")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.TRAINING.GPU_ID)

        cudnn.benchmark = True
        torch.cuda.manual_seed_all(cfg.TRAINING.SEED)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(cfg.TRAINING.SEED)

    # create logging directory

    if not os.path.isdir(LOG_DIR):  os.makedirs(LOG_DIR)
    if not os.path.isdir(MODELS_DIR):  os.makedirs(MODELS_DIR)

    # set random seeds
    random.seed(cfg.TRAINING.SEED)
    np.random.seed(cfg.TRAINING.SEED)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
