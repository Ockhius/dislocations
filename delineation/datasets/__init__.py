import torch
from delineation.datasets.dislocation_dataset import DislocationDataset


def make_data_loader(cfg, save_representations= False):

    num_workers = cfg.TRAINING.NUM_WORKERS

    if (cfg.TRAINING.TRAINING_SET == 'us3d'\
            or cfg.TRAINING.TRAINING_SET == 'dislocations') \
            and save_representations:

        train_set = DislocationDataset(cfg, train=True, save_representations = True)
        val_set   = DislocationDataset(cfg, train=False, save_representations = True)

    elif cfg.TRAINING.TRAINING_SET == 'us3d'\
            or cfg.TRAINING.TRAINING_SET == 'dislocations':

        train_set = DislocationDataset(cfg, train=True, save_representations = False)
        val_set   = DislocationDataset(cfg, train=False, save_representations = False)

    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=cfg.TRAINING.BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=cfg.TEST.BATCH_SIZE,
                                             shuffle=False,
                                             num_workers=num_workers)

    return train_loader, val_loader