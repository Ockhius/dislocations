from delineation.models.unet import UNet
from delineation.models.scmnet import SCMNET, GC_NET
from delineation.models.scmnet_seg import SCMNET_SEG
from delineation.models.scmnet_seg_joint import SCMNET_SEG_JOINT
import torch
from torch import nn

import segmentation_models_pytorch as smp

class SegmentationModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        if 'unet' and 'resnet' in cfg.TRAINING.MODEL:
            print(cfg.TRAINING.MODEL)
            self.model = smp.Unet('resnet34', classes=1, in_channels=cfg.TRAINING.NUM_CHANNELS)
            self.model_type = 'unet_lib'

        elif 'unet' and 'efficient' in cfg.TRAINING.MODEL:
            print('EfficientNet-b0')
            self.model = smp.Unet('efficientnet-b0', classes=1, in_channels=cfg.TRAINING.NUM_CHANNELS)
            self.model_type = 'unet_lib'

        else:
            print('Baseline')
            self.model = UNet(cfg.TRAINING.NUM_CHANNELS)
            self.model_type = 'baseline'

    def forward(self, input):

        if self.model_type == 'baseline':
            return self.model(input)

        x = self.model.encoder(input)
        seg_map = self.model.decoder(*x)
        x = self.model.segmentation_head(seg_map)

        return seg_map, x

def build_segmentation_model(cfg):
    model = SegmentationModel(cfg).cuda()
    print(model)
    return model


def build_model(cfg, save_representations=False):

    if cfg.TRAINING.MODEL == 'unet':
        model = build_segmentation_model(cfg)

       # model = UNet(cfg.TRAINING.NUM_CHANNELS)
    elif cfg.TRAINING.MODEL == 'scmnet':
        model = SCMNET(cfg.TRAINING.NUM_CHANNELS, 1, [cfg.TRAINING.RESBLOCK_NUM, cfg.TRAINING.DISPBLOCK_NUM],
                       cfg.TRAINING.MAXDISP, cfg.TRAINING.DISPSPACE)
    elif cfg.TRAINING.MODEL == 'scmnet_seg_joint':
        model = SCMNET_SEG_JOINT(cfg.TRAINING.NUM_CHANNELS, [1, 1],
                                 [cfg.TRAINING.RESBLOCK_NUM, cfg.TRAINING.DISPBLOCK_NUM],
                                 cfg.TRAINING.MAXDISP, cfg.TRAINING.DISPSPACE)

    if (cfg.TRAINING.RESUME !='' and cfg.TRAINING.RESUME!=','):
        print(cfg.TRAINING.RESUME)

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
        if torch.cuda.device_count() > 1:
            print("Using %d GPUs" % torch.cuda.device_count())
            model = torch.nn.DataParallel(model)
        model.cuda()

    return model


def build_model_list(cfg, is_evaluate):
    model_names = cfg.TRAINING.MODEL.split(',')
    model_weights = cfg.TEST.MODEL_WEIGHTS.split(',')
    model_resume = cfg.TRAINING.RESUME.split(',')
    model_list = []
    for idx, model_n in enumerate(model_names):
        model_n = model_n.strip()
        print(model_n)

        if 'unet' in model_n:

              print(model_n+'-effnet')
              model = SegmentationModel(cfg)

        elif model_n == 'scmnet':

           # model = GC_NET(cfg.TRAINING.MAXDISP, 64, 1, disp_space='two-sided')

            model = SCMNET(16, 1, [cfg.TRAINING.RESBLOCK_NUM, cfg.TRAINING.DISPBLOCK_NUM],
                            cfg.TRAINING.MAXDISP, cfg.TRAINING.DISPSPACE)
        elif model_n == 'scmnet_seg':

            model = SCMNET_SEG(cfg.TRAINING.NUM_CHANNELS, 64, 1,
                               [cfg.TRAINING.RESBLOCK_NUM, cfg.TRAINING.DISPBLOCK_NUM],
                               cfg.TRAINING.MAXDISP, cfg.TRAINING.DISPSPACE)

        if len(model_resume[idx])>3:
            state_dict = torch.load(model_resume[idx])
            new_state_dict = {}
            if idx == 0:
                model.load_state_dict(state_dict['state_dict'])

                # for key,value in state_dict['state_dict'].items():
                #     if 'classification_head' not in key:
                #         new_state_dict[key] = value
            # model.load_state_dict(new_state_dict)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_resume[idx], state_dict['epoch']))

        if is_evaluate and len(model_weights[idx])>3:
            state_dict = torch.load(model_weights[idx])
            model.load_state_dict(state_dict['state_dict'])
            print("=> loaded inference checkpoint '{}' (epoch {})"
                  .format(model_weights[idx], state_dict['epoch']))

            model.eval()

        if cfg.TRAINING.CUDA:
            model.cuda()
        model_list.append(model)

    return model_list
