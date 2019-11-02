import torch


def volume_indices(D, B, H, W, device):
    indices = torch.arange(0, D, 1, dtype=torch.float32, requires_grad=False)
    indices = indices.unsqueeze(0).unsqueeze(0).unsqueeze(3).unsqueeze(3)
    indices = indices.expand(B, 1, D, H, W).to(device).contiguous()

    return indices
