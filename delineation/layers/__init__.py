from .loss import *


def make_loss(cfg):
    type = cfg.TRAINING.LOSS
    print(type)
    if type == 'bce':
        return binary_cross_entropy()

    if type == 'warp_only':
        return warp_only()

    if type == 'warp_only_joint':
        return warp_only_joint()

    if type == 'smooth_l1_masked_disparity':
        return smooth_l1_masked_disparity()

    if type == 'smooth_l1_masked_disparity_joint':
        return smooth_l1_masked_disparity_joint()

    if type == 'smooth_l1_disparity_and_edge_warp':
        return smooth_l1_disparity_and_edge_warp()

    if type == 'smooth_l1_masked_disparity_and_bce':
        return smooth_l1_disparity_and_bce()

    if type == 'smooth_l1_disparity_and_edge_warp_joint':
        return smooth_l1_disparity_and_edge_warp_joint()


    return binary_cross_entropy()