import torch


def make_optimizer(cfg, model):

    if cfg.TRAINING.OPTIMIZER == 'sgd':
        optimizer = getattr(torch.optim, cfg.TRAINING.OPTIMIZER)(model.parameters(), momentum=cfg.TRAINING.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.TRAINING.OPTIMIZER)(model.parameters())

    return optimizer
