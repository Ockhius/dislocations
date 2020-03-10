import argparse
import time
import os
import sys
import torch

from delineation.configs.defaults_segmentation import _C as cfg
from delineation.utils.settings import initialize_cuda_and_logging, debug_segmentation_val
from delineation.models import build_model
from delineation.layers import make_loss
from delineation.solver import make_optimizer
from delineation.datasets import make_data_loader
from delineation.logger import make_logger

sys.path.append(".")

def do_validate(model, val_loader, loss_func):

    model.eval()
    total_test_loss = 0
    for batch_idx, (l, r, lgt, rgt, l_name) in enumerate(val_loader):

        with torch.no_grad():
            l, r, lgt, rgt = l.cuda(), r.cuda(), lgt.cuda(), rgt.cuda()

            seg_l, seg_rep_l = model(l)
            seg_r, seg_rep_r = model(r)

            loss = loss_func(seg_l, lgt, seg_r, rgt)
            total_test_loss += loss.item()

            seg_l, lgt, l = seg_l.detach(), lgt.detach(), l.detach()
            seg_l, seg_r = torch.sigmoid(seg_l) >0.5, torch.sigmoid(seg_r)>0.5

            for i in range(0, len(seg_l)):
                debug_segmentation_val(l[i],
                                  seg_l[i],
                                   lgt[i],
                                   os.path.join(cfg.LOGGING.LOG_DIR, 'check_segmentation_'+str(batch_idx)+'_'+str(i)+'.png'))


    model.train()

    return total_test_loss/ len(val_loader)

def do_train(

        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_func,
        logger):


        start_full_time = time.time()
        model.train()

        start_epoch, end_epoch = cfg.TRAINING.START_EPOCH, cfg.TRAINING.EPOCHS

        for epoch in range(start_epoch, end_epoch + 1):
            print('This is %d-th epoch' % (epoch))
            total_train_loss = 0
        #   adjust_learning_rate(optimizer, epoch)

            for batch_idx, (l, r, lgt, rgt, l_name) in enumerate(train_loader):
                start_time = time.time()

                optimizer.zero_grad()

                l, r, lgt, rgt = l.cuda(), r.cuda(), lgt.cuda(), rgt.cuda()

                seg_l, seg_rep_l = model(l)
                seg_r, seg_rep_r = model(r)

                loss = loss_func(seg_l, lgt, seg_r, rgt)

                loss.backward()
                optimizer.step()

                print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, loss, time.time() - start_time))
                total_train_loss += loss

            print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(train_loader)))

            if epoch % cfg.LOGGING.LOG_INTERVAL == 0:
                total_test_loss = do_validate(model, val_loader, loss_func)
                logger.log_string('test loss for epoch {} : {}\n'.format(epoch, total_test_loss))
                print('epoch %d total test loss = %.3f' % (epoch, total_test_loss))

            if epoch % cfg.TRAINING.SAVE_MODEL_STEP == 0:

                savefilename = os.path.join(cfg.TRAINING.MODEL_DIR, str(epoch) + '_segmentor.tar')

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
        file_logger, tb_logger = make_logger(cfg)

        do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_func,
        file_logger)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Segmentation training")

    parser.add_argument(
        "--config_file", default="/cvlabsrc1/cvlab/datasets_anastasiia/dislocations/dislocations/delineation/configs/blood_vessels_segmentation.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)

    initialize_cuda_and_logging(cfg)
    train(cfg)