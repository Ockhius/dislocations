import argparse
import time
import os
import sys
import torch
import torch.nn.functional as F

from delineation.configs.defaults_segmentation import _C as cfg
from delineation.utils import settings, cost_volume_helpers
from delineation.models import build_model_list
from delineation.layers import make_loss
from delineation.solver import make_optimizer
from delineation.datasets import make_data_loader
from delineation.logger import make_logger
from delineation.utils.settings import evaluate_results

sys.path.append(".")

def do_validate(epoch, seg_model, model, val_loader, loss_func, tf_logger):

    seg_model.eval()
    model.eval()
    total_test_loss = 0
    pix1_err_m, pix3_err_m, pix5_err_m, epe_m, count = 0, 0, 0, 0, 0


    for batch_idx, (l, r, lgt, rgt, dlgt, l_name) in enumerate(val_loader):
        indices = cost_volume_helpers.volume_indices(2 * cfg.TRAINING.MAXDISP, len(l),
                                                     cfg.TRAINING.HEIGHT, cfg.TRAINING.WIDTH, _device)

        with torch.no_grad():
            l, r, lgt, rgt, dlgt = l.to(_device), r.to(_device), lgt.to(_device), rgt.to(_device), dlgt.to(_device)

            l_seg, l_segmap = seg_model(l)
            r_seg, r_segmap = seg_model(r)

            dl_scores = model(l_segmap, r_segmap)
            dl = F.softmax(-dl_scores, 2)
            dl = torch.sum(dl.mul(indices), 2) - cfg.TRAINING.MAXDISP

            loss = loss_func(dl, rgt, dlgt, lgt)
            total_test_loss += loss.item()

            for i in range(0, len(dlgt)):
                pix1_err, pix3_err, pix5_err, epe = evaluate_results(dlgt[i], dl[i], lgt[i])

                pix1_err_m += pix1_err
                pix3_err_m += pix3_err
                pix5_err_m += pix5_err
                epe_m += epe
                count += 1

    values = pix1_err_m / count, pix3_err_m / count, pix5_err_m / count, epe_m / count
    tf_logger.add_scalars_to_tensorboard('Test', epoch, epoch, total_test_loss / len(val_loader), values)
    print('Mean per dataset: {}, {}, {}, {}'.format(pix1_err_m / count, pix3_err_m / count, pix5_err_m / count, epe_m / count))

    model.train()

    return total_test_loss / len(val_loader)


def do_train(cfg, seg_model, model, train_loader, val_loader, optimizer, loss_func, logger, tf_logger):
    start_full_time = time.time()
    seg_model.eval()
    model.train()

    start_epoch, end_epoch = cfg.TRAINING.START_EPOCH, cfg.TRAINING.EPOCHS

    iter_count = 0
    for epoch in range(start_epoch, end_epoch + 1):
        print('This is %d-th epoch' % epoch)
        total_train_loss = 0

        for batch_idx, (l, r, lgt, rgt, dlgt, l_name) in enumerate(train_loader):

            indices = cost_volume_helpers.volume_indices(2 * cfg.TRAINING.MAXDISP, len(l),
                                                         cfg.TRAINING.HEIGHT, cfg.TRAINING.WIDTH, _device)

            start_time = time.time()


            l, r, lgt, rgt, dlgt = l.to(_device), r.to(_device), lgt.to(_device), rgt.to(_device), dlgt.to(_device)

            l_seg, l_segmap = seg_model(l)
            r_seg, r_segmap = seg_model(r)

            optimizer.zero_grad()

            dl_scores = model(l_segmap, r_segmap)
            dl = F.softmax(-dl_scores, 2)

            dl = torch.sum(dl.mul(indices), 2) - cfg.TRAINING.MAXDISP
            loss = loss_func(dl, rgt, dlgt, lgt)

            loss.backward()

            optimizer.step()

            print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, loss.item(), time.time() - start_time))
            total_train_loss += float(loss)

            tf_logger.add_loss_to_tensorboard('Train/Loss', loss.item(),iter_count )
            iter_count+=1

        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(train_loader)))

        if epoch % cfg.LOGGING.LOG_INTERVAL == 0:
            total_test_loss = do_validate(epoch, seg_model, model, val_loader, loss_func, tf_logger)
            logger.log_string('test loss for epoch {} : {}\n'.format(epoch, total_test_loss))
            print('epoch %d total test loss = %.3f' % (epoch, total_test_loss))

        if epoch % cfg.TRAINING.SAVE_MODEL_STEP == 0:
            savefilename = os.path.join(cfg.TRAINING.MODEL_DIR, str(epoch) + '_scmnet_light.tar')

            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss / len(train_loader.dataset),
            }, savefilename)

            print('model is saved: {} - {}'.format(epoch, savefilename))

    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))


def train(cfg):
    # create dataset
    train_loader, val_loader = make_data_loader(cfg)

    # create model
    seg_model, model = build_model_list(cfg, False)

    # create optimizer
    optimizer = make_optimizer(cfg, model)

    # create loss
    loss_func = make_loss(cfg)

    # create logger
    logger, tf_logger = make_logger(cfg)

    do_train(
        cfg,
        seg_model,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_func,
        logger, tf_logger)


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

    train(cfg)