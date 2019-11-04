import torch


def cost_volume(ref, target, disp_space='two-sided', maxdisp=32):
    if disp_space == 'two-sided':
        lowest = - maxdisp
        highest = maxdisp
        disp_size = 2 * maxdisp
    else:
        lowest = 0
        highest = maxdisp
        disp_size = maxdisp

    # N x 2F x D/2 x H/2 x W/2
    cost = torch.FloatTensor(ref.size()[0],
                             ref.size()[1] * 2,
                             disp_size,
                             ref.size()[2],
                             ref.size()[3]).zero_().cuda()

    # reference is the left image, target is the right image. IL (+) DL = IR
    for i, k in enumerate(range(lowest, highest)):
        if k == 0:
            cost[:, :ref.size()[1], i, :, :] = ref
            cost[:, ref.size()[1]:, i, :, :] = target
        elif k < 0:
            cost[:, :ref.size()[1], i, :, -k:] = ref[:, :, :, -k:]
            cost[:, ref.size()[1]:, i, :, -k:] = target[:, :, :, :k]  # target shifted to right over ref
        else:
            cost[:, :ref.size()[1], i, :, :-k] = ref[:, :, :, :-k]
            cost[:, ref.size()[1]:, i, :, :-k] = target[:, :, :, k:]  # target shifted to left over ref

    return cost.contiguous()