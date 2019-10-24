import argparse
import numpy as np
import time
import os
import sys
import torch
import matplotlib.pyplot as plt

from delineation.configs.defaults_segmentation import _C as cfg
from delineation.utils import settings
from delineation.models import build_model
from delineation.layers import make_loss
from delineation.datasets import make_data_loader

sys.path.append(".")


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


def do_inference(val_loader, model, loss_func):
    model.eval()
    total_test_loss = 0
    for batch_idx, (l, r, lgt, rgt, l_name) in enumerate(val_loader):
        with torch.no_grad():
            l, r, lgt, rgt = l.cuda(), r.cuda(), lgt.cuda(), rgt.cuda()

            seg_l, seg_rep_l = model(l)
            seg_r, seg_rep_r = model(r)

            loss = loss_func(seg_l, lgt, seg_r, rgt)
            total_test_loss += float(loss)

            seg_l, seg_r = torch.sigmoid(seg_l) > 0.5, torch.sigmoid(seg_r) > 0.5
            for i in range(0, len(seg_l)):
                debug_segmentation_val(l[i], seg_l[i], lgt[i],
                                       os.path.join(cfg.LOGGING.LOG_DIR,
                                                    'check_segmentation_'+str(batch_idx)+'_'+str(i)+'.png'))


def inference(cfg):
    # create dataset
    train_loader, val_loader = make_data_loader(cfg)

    # create model
    model = build_model(cfg, True)

    # create loss
    loss_func = make_loss(cfg)

    do_inference(val_loader, model, loss_func)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Dislocation Segmentation training")

    parser.add_argument(
        "--config_file", default="delineation/configs/dislocation_segmentation_inference_home.yml", help="path to config file",
        type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)

    settings.initialize_cuda_and_logging(cfg)

    inference(cfg)