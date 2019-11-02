from delineation.models.unet_small import UNet_small
from delineation.models.unet import UNet
from delineation.models.unet_even_smaller import UNet_smaller
from delineation.models.tinynet import TinyNet
from delineation.models.scmnet import SCMNET
import torch


def build_model(cfg, save_representations=False):

    #model = UNet(cfg.TRAINING.NUM_CHANNELS)
    #model = UNet_smaller(cfg.TRAINING.NUM_CHANNELS)
    #model = TinyNet(cfg.TRAINING.NUM_CHANNELS)

    if cfg.TRAINING.MODEL == 'unet':
        model = UNet(cfg.TRAINING.NUM_CHANNELS)
    elif cfg.TRAINING.MODEL == 'scmnet':
        model = SCMNET(cfg.TRAINING.NUM_CHANNELS, 1, [cfg.TRAINING.RESBLOCK_NUM, cfg.TRAINING.DISPBLOCK_NUM],
                       cfg.TRAINING.MAXDISP, cfg.TRAINING.DISPSPACE)

    if cfg.TRAINING.RESUME !='':
        print('Loading chosen state for segmentor...')
        state_dict = torch.load(cfg.TRAINING.RESUME)
        model.load_state_dict(state_dict['state_dict'])

    if save_representations:
        state_dict = torch.load(cfg.TEST.MODEL_WEIGHTS)
        model.load_state_dict(state_dict['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(cfg.TEST.MODEL_WEIGHTS, state_dict['epoch']))

        model.eval()

    if cfg.TRAINING.CUDA:
        model.cuda()

    return model