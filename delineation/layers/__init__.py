from .loss import binary_cross_entropy


def make_loss(cfg):
    type = cfg.TRAINING.LOSS

    if type == 'bce':
        return binary_cross_entropy()

    return binary_cross_entropy()