import argparse
import sys
import torch
import torch.nn.functional as F
import yaml

sys.path.append(".")

from delineation.utils.settings import evaluate_results
from delineation.configs.defaults_segmentation import _C as cfg
from delineation.utils import settings, cost_volume_helpers
from delineation.models import build_model_list
from delineation.datasets import make_data_loader

def do_inference(val_loader, seg_model, model):

    seg_model.eval()
    model.eval()

    pix1_err_m, pix3_err_m, pix5_err_m, epe_m, count = 0, 0, 0, 0, 0

    for batch_idx, (l_aug, r_aug, l_gt_aug, r_gt_aug, l, r, lgt, rgt, dlgt, l_name) in enumerate(val_loader):
        indices = cost_volume_helpers.volume_indices(2 * cfg.TRAINING.MAXDISP, len(l),
                                                     cfg.TRAINING.HEIGHT, cfg.TRAINING.WIDTH, _device)

        with torch.no_grad():
            l, r, lgt, rgt, dlgt = l.to(_device), r.to(_device), lgt.to(_device), rgt.to(_device), dlgt.to(_device)


            l_segmap, l_seg = seg_model(l)
            r_segmap, r_seg = seg_model(r)

            dl_scores = model(l_segmap, r_segmap)

            dl_ = F.softmax(-dl_scores, 2)
            dl = torch.sum(dl_.mul(indices), 2) - cfg.TRAINING.MAXDISP

            for i in range(0, len(dlgt)):
                pix1_err, pix3_err, pix5_err, epe = evaluate_results(dlgt[i], dl[i], lgt[i])

                pix1_err_m += pix1_err
                pix3_err_m += pix3_err
                pix5_err_m += pix5_err
                epe_m += epe
                count += 1

    print('Mean per dataset: {}, {}, {}, {}'.format(pix1_err_m / count, pix3_err_m / count, pix5_err_m / count,
                                                    epe_m / count))

    return pix1_err_m / count, pix3_err_m / count, pix5_err_m / count,  epe_m / count

def inference(cfg, cfg_aug):

    # create dataset
    train_loader, val_loader, test_loader = make_data_loader(cfg, cfg_aug)

    # create model
    seg_model, model = build_model_list(cfg, True)

    pix1_val, pix3_val, pix5_val, epe_val = do_inference(val_loader, seg_model, model)
    pix1, pix3, pix5, epe = do_inference(test_loader, seg_model, model)

    print("Validation: Pix1 error: {}, Pix3 error: {}, Pix5 error:{}, EPE: {} ".format(pix1_val, pix3_val, pix5_val, epe_val))
    print("Test: Pix1 error: {}, Pix3 error: {}, Pix5 error:{}, EPE: {} ".format(pix1, pix3, pix5, epe))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Dislocation Segmentation training")

    parser.add_argument(
        "--config_file", default="/cvlabsrc1/cvlab/datasets_anastasiia/dislocations/dislocations/delineation/configs/32/dislocation_matching_disp_and_var_joint_small_dataset.yml", help="path to config file",
        type=str
    )

    parser.add_argument('--path_ymlfile', type=str,default='delineation/configs/aug.yml', help='Path to yaml file.')

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)

    with open(args.path_ymlfile, 'r') as ymlfile:
        cfg_aug = yaml.load(ymlfile)

    _device = settings.initialize_cuda_and_logging(cfg)  # '_device' is GLOBAL VAR

    inference(cfg, cfg_aug)