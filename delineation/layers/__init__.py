from .loss import binary_cross_entropy, smooth_l1_masked_dispartiy, smooth_l1_disparity_and_edge_warp


def make_loss(cfg):
    type = cfg.TRAINING.LOSS

    if type == 'bce':
        return binary_cross_entropy()

    if type == 'smooth_l1_masked_disparity':
        return smooth_l1_masked_dispartiy()

    if type == 'smooth_l1_diparity_and_edge_warp':
        return smooth_l1_disparity_and_edge_warp()

    return binary_cross_entropy()