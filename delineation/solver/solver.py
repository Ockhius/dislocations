import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

def make_optimizer(cfg, model):

    if 'joint' in cfg.TRAINING.LOSS:
        print('Create joint optimizer')
        params = list(model[0].parameters()) + list(model[1].parameters())
        optimizer = getattr(torch.optim, cfg.TRAINING.OPTIMIZER)(params, lr=cfg.TRAINING.BASE_LR, betas=(0.9, 0.999))
        return optimizer

    if cfg.TRAINING.OPTIMIZER == 'sgd':
        optimizer = getattr(torch.optim, cfg.TRAINING.OPTIMIZER)(model.parameters(), momentum=cfg.TRAINING.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.TRAINING.OPTIMIZER)(model.parameters(), lr=cfg.TRAINING.BASE_LR, betas=(0.9, 0.999))

    return optimizer


def make_scheduler(cfg, optimizer):

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    return scheduler