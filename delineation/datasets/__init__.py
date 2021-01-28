import torch
from delineation.datasets.dislocation_dataset import DislocationDataset
from delineation.datasets.dislocation_matching_dataset import MatchingDislocationsDataset
from delineation.datasets.dislocation_matching_dataset_aug import MatchingDislocationsDataset as MatchingDislocationsDatasetAug
from delineation.datasets.dislocation_segmentation_only_dataset import DislocationsSegmentationDataset

def make_data_loader(cfg, cfg_aug, save_representations= False):

    num_workers = cfg.TRAINING.NUM_WORKERS

    test_set, test_loader = None, None

    if (cfg.TRAINING.TRAINING_SET == 'dislocations') and save_representations:

        train_set = DislocationDataset(cfg, train=True, save_representations = True)
        val_set   = DislocationDataset(cfg, train=False, save_representations = True)

    elif cfg.TRAINING.TRAINING_SET == 'dislocations':

        train_set = DislocationDataset(cfg, train=True, save_representations = False)
        val_set   = DislocationDataset(cfg, train=False, save_representations = False)


    elif cfg.TRAINING.TRAINING_SET == 'dislocations_matching' and 'joint' in cfg.TRAINING.LOSS:

        train_set = MatchingDislocationsDatasetAug(cfg, cfg_aug, split='train')
        val_set = MatchingDislocationsDatasetAug(cfg, cfg_aug, split='val')
        test_set = MatchingDislocationsDatasetAug(cfg, cfg_aug, split='test')

    elif cfg.TRAINING.TRAINING_SET == 'dislocations_matching':

        train_set = MatchingDislocationsDataset(cfg, cfg_aug, train=True)
        val_set = MatchingDislocationsDataset(cfg, cfg_aug, train=False)

    elif cfg.TRAINING.TRAINING_SET == 'dislocations_segmentation_only':
        train_set = DislocationsSegmentationDataset(cfg, train=True, save_representations = False)
        val_set   = DislocationsSegmentationDataset(cfg, train=False, save_representations = False)

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
    if test_set is not None:
        test_loader = torch.utils.data.DataLoader(test_set,
                                                 batch_size=cfg.TEST.BATCH_SIZE,
                                                 shuffle=False,
                                                 num_workers=num_workers)

    return train_loader, val_loader, test_loader