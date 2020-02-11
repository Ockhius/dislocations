import argparse
from tqdm import tqdm
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F

from delineation.configs.defaults_segmentation import _C as cfg
from delineation.utils import settings, cost_volume_helpers
from delineation.models import build_model, build_model_list
from delineation.layers import make_loss
from delineation.datasets import make_data_loader
from delineation.utils.settings import evaluate_results
sys.path.append(".")


def do_inference(val_loader, seg_model, model, loss_func):
    model.eval()
    seg_model.eval()
    total_test_loss = 0
    total_epe = 0
    pix1_err_m, pix3_err_m, pix5_err_m, epe_m, count = 0, 0, 0, 0, 0

    for batch_idx, (l, r, lgt, rgt, dlgt, l_name) in enumerate(val_loader):
        with torch.no_grad():

            indices = cost_volume_helpers.volume_indices(2 * cfg.TRAINING.MAXDISP, len(l),
                                                         cfg.TRAINING.HEIGHT, cfg.TRAINING.WIDTH, _device)
            l, r, lgt, rgt, dlgt = l.to(_device), r.to(_device), lgt.to(_device), rgt.to(_device), dlgt.to(_device)

            with torch.no_grad():
                l_seg, l_segmap = seg_model(l)
                r_seg, r_segmap = seg_model(r)

            left_scores = model(l_segmap, r_segmap)
            left_disp_pred = F.softmax(-left_scores, 2)
            dl = torch.sum(left_disp_pred.mul(indices), 2) - 32

            print(dl.shape)

            loss = loss_func(dl, rgt, dlgt, lgt)
            total_test_loss += float(loss)
            epe = np.mean(np.abs((dl[lgt > 0] - dlgt.unsqueeze(1)[lgt > 0]).cpu().numpy()))
            print(epe)
            total_epe += float(epe)

            for i in range(0, len(dlgt)):
                print(l_name[i])
                pix1_err, pix3_err, pix5_err, epe = evaluate_results(dlgt[i], dl[i], lgt[i])

                pix1_err_m += pix1_err
                pix3_err_m += pix3_err
                pix5_err_m += pix5_err
                epe_m += epe
                count += 1

                print(dl[i].max())
                print(dl[i].min())
                print(dlgt[i].max())
                print(dlgt[i].min())


            lgt = lgt.cpu().numpy()
            dl = dl.detach().permute(0, 2, 3, 1).cpu().numpy()

            dlgt = dlgt.unsqueeze(1).detach().permute(0, 2, 3, 1).cpu().numpy()
            for i in tqdm(range(len(dl))):
                vol = cost_volume_helpers.back_project_numpy(lgt[i,0,:,:], dl[i,:,:,:], cfg.TRAINING.MAXDISP, mode='two-sided')
                vol_gt = cost_volume_helpers.back_project_numpy(lgt[i,0,:,:], dlgt[i,:,:,:], cfg.TRAINING.MAXDISP, mode='two-sided')

                for idx, rotation_angle in enumerate(range(-20, 20, 1)):
                    cost_volume_helpers.visualize_volume(l_name[i], vol[0, :, :, :],
                                                         rotation_angle,
                                                         cfg.LOGGING.LOG_DIR,
                                                         mode='scatter',
                                                         save_ext=idx,
                                                         plot=False)

                    cost_volume_helpers.visualize_volume(l_name[i], vol_gt[0, :, :, :],
                                                         rotation_angle,
                                                         cfg.LOGGING.LOG_DIR+'_gt',
                                                         mode='scatter',
                                                         save_ext=idx,
                                                         plot=False)

                torch.cuda.empty_cache()

    print('Mean per dataset: {}, {}, {}, {}'.format(pix1_err_m / count, pix3_err_m / count, pix5_err_m / count, epe_m / count))

    return total_test_loss / len(val_loader)


def inference(cfg):
    # create dataset
    train_loader, val_loader = make_data_loader(cfg)

    # create model
    seg_model, model = build_model_list(cfg, True)

    # create loss
    loss_func = make_loss(cfg)

    total_score = do_inference(val_loader, seg_model, model, loss_func)

    print('Total scode = {}'.format(total_score))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Dislocation Segmentation training")

    parser.add_argument(
        "--config_file", default="delineation/configs/dislocation_matching.yml", help="path to config file",
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