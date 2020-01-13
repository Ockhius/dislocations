from .loss import *


def make_loss(cfg):
    type = cfg.TRAINING.LOSS
    print(type)
    if type == 'bce':
        return binary_cross_entropy()

    if type == 'smooth_l1_masked_disparity':
        return smooth_l1_masked_dispartiy()

    if type == 'smooth_l1_disparity_and_edge_warp':
        return smooth_l1_disparity_and_edge_warp()

    if type == 'smooth_l1_masked_disparity_and_bce':
        return smooth_l1_disparity_and_bce()

    return binary_cross_entropy()