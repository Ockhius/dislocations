from delineation.models.unet import UNet
from delineation.models.scmnet import SCMNET, GC_NET
from delineation.models.scmnet_seg import SCMNET_SEG
from delineation.models.scmnet_seg_joint import SCMNET_SEG_JOINT
import torch


def build_model(cfg, save_representations=False):
    if cfg.TRAINING.MODEL == 'unet':
        print(cfg.TRAINING.MODEL)

        model = UNet(cfg.TRAINING.NUM_CHANNELS)
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


def build_model_list(cfg, save_representations):
    model_names = cfg.TRAINING.MODEL.split(',')
    model_weights = cfg.TEST.MODEL_WEIGHTS.split(',')
    model_resume = cfg.TRAINING.RESUME.split(',')
    model_list = []
    for idx, model_n in enumerate(model_names):
        model_n = model_n.strip()
        print(model_n)

        if model_n == 'unet':
            model = UNet(cfg.TRAINING.NUM_CHANNELS)
        elif model_n == 'scmnet':

            model = GC_NET(cfg.TRAINING.MAXDISP, 64, 1, disp_space='two-sided')

            # model = SCMNET(64, 1, [cfg.TRAINING.RESBLOCK_NUM, cfg.TRAINING.DISPBLOCK_NUM],
            #                 cfg.TRAINING.MAXDISP, cfg.TRAINING.DISPSPACE)
        elif model_n == 'scmnet_seg':

            model = SCMNET_SEG(cfg.TRAINING.NUM_CHANNELS, 64, 1,
                               [cfg.TRAINING.RESBLOCK_NUM, cfg.TRAINING.DISPBLOCK_NUM],
                               cfg.TRAINING.MAXDISP, cfg.TRAINING.DISPSPACE)

        if len(model_resume[idx])>3:
            state_dict = torch.load(model_resume[idx])
            model.load_state_dict(state_dict['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_resume[idx], state_dict['epoch']))

        if save_representations and len(model_weights[idx])>3:
            state_dict = torch.load(model_weights[idx])
            model.load_state_dict(state_dict['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_weights[idx], state_dict['epoch']))

            model.eval()

        if cfg.TRAINING.CUDA:
            model.cuda()
        model_list.append(model)

    return model_list
