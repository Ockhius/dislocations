import torch

def make_optimizer(cfg, model):

    if 'joint' in cfg.TRAINING.LOSS:
        params = list(model[0].parameters()) + list(model[1].parameters())
        optimizer = getattr(torch.optim, cfg.TRAINING.OPTIMIZER)(params, lr=cfg.TRAINING.BASE_LR, betas=(0.9, 0.999))
        return optimizer

    if cfg.TRAINING.OPTIMIZER == 'sgd':
        optimizer = getattr(torch.optim, cfg.TRAINING.OPTIMIZER)(model.parameters(), momentum=cfg.TRAINING.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.TRAINING.OPTIMIZER)(model.parameters(), lr=cfg.TRAINING.BASE_LR, betas=(0.9, 0.999))

    return optimizer
