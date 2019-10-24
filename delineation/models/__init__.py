from delineation.models.unet_small import UNet_small
from delineation.models.unet import UNet
from delineation.models.unet_even_smaller import UNet_smaller
from delineation.models.tinynet import TinyNet
import torch


def build_model(cfg, save_representations=False):

    model = UNet(cfg.TRAINING.NUM_CHANNELS)
    #model = UNet_smaller(cfg.TRAINING.NUM_CHANNELS)
    #model = TinyNet(cfg.TRAINING.NUM_CHANNELS)

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