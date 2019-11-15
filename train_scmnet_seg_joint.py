import argparse
import time
import os
import sys
import torch
import torch.nn.functional as F

from delineation.configs.defaults_segmentation import _C as cfg
from delineation.utils import settings, cost_volume_helpers
from delineation.models import build_model
from delineation.layers import make_loss
from delineation.solver import make_optimizer
from delineation.datasets import make_data_loader
from delineation.logger import make_logger

sys.path.append(".")


def do_validate(model, val_loader, loss_func):

    model.eval()
    total_test_loss = 0

    indices = cost_volume_helpers.volume_indices(2 * cfg.TRAINING.MAXDISP, cfg.TEST.BATCH_SIZE,
                                                 cfg.TRAINING.HEIGHT, cfg.TRAINING.WIDTH, _device)

    for batch_idx, (l, r, lgt, _, dlgt, l_name) in enumerate(val_loader):

        with torch.no_grad():
            l, r, lgt, dlgt = l.to(_device), r.to(_device), lgt.to(_device), dlgt.to(_device)

            dl_scores, sl = model(l, r)
            dl = F.softmax(-dl_scores, 2)
            dl = torch.sum(dl.mul(indices[:len(l), :, :, :]), 2) - cfg.TRAINING.MAXDISP  # CHECK HERE AGAIN !

            loss, _, _ = loss_func(dl, dlgt, sl, lgt)

            total_test_loss += float(loss)

    model.train()

    return total_test_loss / len(val_loader)


def do_train(cfg, model, train_loader, val_loader, optimizer, loss_func, logger):
    start_full_time = time.time()
    model.train()

    start_epoch, end_epoch = cfg.TRAINING.START_EPOCH, cfg.TRAINING.EPOCHS

    indices = cost_volume_helpers.volume_indices(2*cfg.TRAINING.MAXDISP, cfg.TRAINING.BATCH_SIZE,
                                                 cfg.TRAINING.HEIGHT, cfg.TRAINING.WIDTH, _device)

    for epoch in range(start_epoch, end_epoch + 1):
        print('This is %d-th epoch' % epoch)
        total_train_loss = 0

        for batch_idx, (l, r, lgt, _, dlgt, l_name) in enumerate(train_loader):

            start_time = time.time()

            optimizer.zero_grad()

            l, r, lgt, dlgt = l.to(_device), r.to(_device), lgt.to(_device), dlgt.to(_device)

            dl_scores, sl = model(l, r)
            dl = F.softmax(-dl_scores, 2)
            dl = torch.sum(dl.mul(indices[:len(l), :, :, :]), 2) - cfg.TRAINING.MAXDISP

            loss, loss_dl, loss_sl = loss_func(dl, dlgt, sl, lgt)

            loss.backward()
            optimizer.step()

            print('Iter %d '
                  'training loss = %.3f '
                  'disp loss = %.3f '
                  'seg loss = %.3f '
                  'time = %.2f' % (batch_idx, loss.item(), loss_dl.item(), loss_sl.item(), time.time() - start_time))
            total_train_loss += float(loss)

        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(train_loader)))

        if epoch % cfg.LOGGING.LOG_INTERVAL == 0:
            total_test_loss = do_validate(model, val_loader, loss_func)
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
    model = build_model(cfg)

    # create optimizer
    optimizer = make_optimizer(cfg, model)

    # create loss
    loss_func = make_loss(cfg)

    # create logger
    logger = make_logger(cfg)

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_func,
        logger)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Dislocation Segmentation training")

    parser.add_argument(
        "--config_file", default="delineation/configs/dislocation_matching_seg_joint_home.yml", help="path to config file",
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