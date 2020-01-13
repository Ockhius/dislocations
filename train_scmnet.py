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

sys.path.append(".")


def do_validate(seg_model, model, val_loader, loss_func):

    seg_model.eval()
    model.eval()
    total_test_loss = 0


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

            loss = loss_func(dl, dlgt, lgt)

            total_test_loss += float(loss)

    model.train()

    return total_test_loss / len(val_loader)


def do_train(cfg, seg_model, model, train_loader, val_loader, optimizer, loss_func, logger):
    start_full_time = time.time()
    seg_model.eval()
    model.train()

    start_epoch, end_epoch = cfg.TRAINING.START_EPOCH, cfg.TRAINING.EPOCHS


    for epoch in range(start_epoch, end_epoch + 1):
        print('This is %d-th epoch' % epoch)
        total_train_loss = 0

        for batch_idx, (l, r, lgt, rgt, dlgt, l_name) in enumerate(train_loader):

            indices = cost_volume_helpers.volume_indices(2 * cfg.TRAINING.MAXDISP, len(l),
                                                         cfg.TRAINING.HEIGHT, cfg.TRAINING.WIDTH, _device)

            start_time = time.time()

            optimizer.zero_grad()

            l, r, lgt, rgt, dlgt = l.to(_device), r.to(_device), lgt.to(_device), rgt.to(_device), dlgt.to(_device)

            l_seg, l_segmap = seg_model(l)
            r_seg, r_segmap = seg_model(r)

            dl_scores = model(l_segmap, r_segmap)
            dl = F.softmax(-dl_scores, 2)

            dl = torch.sum(dl.mul(indices), 2) - cfg.TRAINING.MAXDISP
            loss = loss_func(dl, dlgt , lgt)

            loss.backward()
            optimizer.step()

            print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, loss.item(), time.time() - start_time))
            total_train_loss += float(loss)

        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(train_loader)))

        if epoch % cfg.LOGGING.LOG_INTERVAL == 0:
            total_test_loss = do_validate(seg_model, model, val_loader, loss_func)
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
    logger = make_logger(cfg)

    do_train(
        cfg,
        seg_model,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_func,
        logger)


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