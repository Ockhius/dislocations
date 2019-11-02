from .loss import binary_cross_entropy, smooth_l1_masked_dispartiy


def make_loss(cfg):
    type = cfg.TRAINING.LOSS

    if type == 'bce':
        return binary_cross_entropy()

    if type == 'smooth_l1_masked_disparity':
        return smooth_l1_masked_dispartiy()

    return binary_cross_entropy()