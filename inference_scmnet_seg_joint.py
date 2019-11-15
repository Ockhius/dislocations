import argparse
from tqdm import tqdm
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from delineation.configs.defaults_segmentation import _C as cfg
from delineation.utils import settings, cost_volume_helpers
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
    total_disp_loss = 0
    total_seg_loss = 0
    total_epe = 0
    indices = cost_volume_helpers.volume_indices(2 * cfg.TRAINING.MAXDISP, cfg.TEST.BATCH_SIZE,
                                                 cfg.TRAINING.HEIGHT, cfg.TRAINING.WIDTH, _device)

    for batch_idx, (l, r, lgt, _, dlgt, l_name) in enumerate(val_loader):
        with torch.no_grad():
            l, r, lgt, dlgt = l.to(_device), r.to(_device), lgt.to(_device), dlgt.to(_device)

            dl_scores, sl = model(l, r)
            dl = F.softmax(-dl_scores, 2)
            dl = torch.sum(dl.mul(indices[:len(l), :, :, :]), 2) - cfg.TRAINING.MAXDISP

            loss, loss_disp, loss_seg = loss_func(dl, dlgt, sl, lgt)
            total_disp_loss += float(loss_disp)
            total_seg_loss += float(loss_seg)

            epe = np.mean(np.abs((dl[lgt > 0] - dlgt.unsqueeze(1)[lgt > 0]).cpu().numpy()))
            print(epe)
            total_epe += float(epe)

            #lgt = lgt.cpu().numpy()
            dl = dl.detach().permute(0, 2, 3, 1).cpu().numpy()
            sl = torch.sigmoid(sl)
            debug_segmentation_val(l[0], sl[0], lgt[0],
                                   os.path.join(cfg.LOGGING.LOG_DIR, 'check_segmentation_' +
                                                str(batch_idx) + '_' + str(batch_idx)+'.png'))


            # dlgt = dlgt.unsqueeze(1).detach().permute(0, 2, 3, 1).cpu().numpy()

            # for i in tqdm(range(len(dl))):
            #     vol = cost_volume_helpers.back_project_numpy(lgt[i,0,:,:], dl[i,:,:,:], cfg.TRAINING.MAXDISP, mode='two-sided')
            #
            #     for idx, rotation_angle in enumerate(range(-40, 40, 1)):
            #         cost_volume_helpers.visualize_volume(l_name[i], vol[0, :, :, :],
            #                                              rotation_angle,
            #                                              cfg.LOGGING.LOG_DIR,
            #                                              mode='scatter',
            #                                              save_ext=idx,
            #                                              plot=False)
            #
            #     torch.cuda.empty_cache()

    return total_disp_loss / len(val_loader), total_epe / len(val_loader), total_seg_loss / len(val_loader)


def inference(cfg):
    # create dataset
    train_loader, val_loader = make_data_loader(cfg)

    # create model
    model = build_model(cfg, True)

    # create loss
    loss_func = make_loss(cfg)

    disp_loss, epe, seg_loss = do_inference(val_loader, model, loss_func)
    print("Total Disp Loss = %.3f, Total Seg Loss = %.3f " % (disp_loss, seg_loss))
    print("EPE = %.3f " % epe)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Dislocation Segmentation training")

    parser.add_argument(
        "--config_file", default="delineation/configs/dislocation_matching_seg_joint_inference_home.yml", help="path to config file",
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