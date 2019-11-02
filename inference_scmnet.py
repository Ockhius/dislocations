import argparse
from tqdm import tqdm
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F

from delineation.configs.defaults_segmentation import _C as cfg
from delineation.utils import settings, cost_volume_helpers
from delineation.models import build_model
from delineation.layers import make_loss
from delineation.datasets import make_data_loader

sys.path.append(".")


def do_inference(val_loader, model, loss_func):

    model.eval()
    total_test_loss = 0
    indices = cost_volume_helpers.volume_indices(2 * cfg.TRAINING.MAXDISP, cfg.TEST.BATCH_SIZE,
                                                 cfg.TRAINING.HEIGHT, cfg.TRAINING.WIDTH, _device)

    for batch_idx, (l, r, lgt, _, dlgt, l_name) in enumerate(val_loader):
        with torch.no_grad():
            l, r, lgt, dlgt = l.to(_device), r.to(_device), lgt.to(_device), dlgt.to(_device)

            dl_scores = model(l, r)
            dl = F.softmax(-dl_scores, 2)
            dl = torch.sum(dl.mul(indices[:len(l), :, :, :]), 2) - cfg.TRAINING.MAXDISP

            loss = loss_func(dl, dlgt, lgt)
            total_test_loss += float(loss)

            print(np.mean(np.abs((dl[lgt > 0] - dlgt.unsqueeze(1)[lgt > 0]).cpu().numpy())))
            print(dl.max().item())
            print(dl.min().item())
            print(dlgt.max().item())
            print(dlgt.min().item())
            print(loss.item())
            lgt = lgt.cpu().numpy()
            dl = dl.detach().permute(0, 2, 3, 1).cpu().numpy()
            # dlgt = dlgt.unsqueeze(1).detach().permute(0, 2, 3, 1).cpu().numpy()
            for i in tqdm(range(len(dl))):
                vol = cost_volume_helpers.back_project_numpy(lgt[i,0,:,:], dl[i,:,:,:], cfg.TRAINING.MAXDISP, mode='two-sided')

                for idx, rotation_angle in enumerate(range(-40, 40, 1)):
                    cost_volume_helpers.visualize_volume(l_name[i], vol[0, :, :, :],
                                                         rotation_angle,
                                                         cfg.LOGGING.LOG_DIR,
                                                         mode='scatter',
                                                         save_ext=idx,
                                                         plot=False)

                torch.cuda.empty_cache()

    return total_test_loss / len(val_loader)


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
        "--config_file", default="delineation/configs/dislocation_matching_inference_home.yml", help="path to config file",
        type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)

    _device = settings.initialize_cuda_and_logging(cfg)  # '_device' is GLOBAL VAR

    inference(cfg)