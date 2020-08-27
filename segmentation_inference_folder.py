from delineation.configs.defaults_segmentation import _C as cfg
from delineation.utils import settings
from delineation.models import build_model
import argparse
import os
from PIL import Image
import numpy as np
import torch
import cv2

def inference(cfg):

    if not os.path.exists(args.input_folder):
        os.makedirs(args.input_folder)

    model = build_model(cfg, True)
    model = model.cuda()

    for img_name in os.listdir(args.input_folder):
        img = np.array(Image.open(os.path.join(args.input_folder, img_name)))
        with torch.no_grad():
            img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0) / 255.0
            segmented_img = model(img.cuda())
            segmented_img = segmented_img.cpu().numpy().squeeze() > 0.1
            cv2.imwrite(os.path.join(args.output_folder, img_name), segmented_img)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Dislocation Segmentation training")

    parser.add_argument(
        "--config_file", default="delineation/configs/dislocation_segmentation_inference_home.yml", help="path to config file",
        type=str
    )

    parser.add_argument(
        "--input_folder", default="/cvlabdata2/cvlab/datasets_anastasiia/Datasets/Dislocations/DislocationsTrainingDataPng/", help="path to config file",
        type=str
    )


    parser.add_argument(
        "--output_folder", default="/cvlabdata2/cvlab/datasets_anastasiia/Datasets/Dislocations/DislocationsTrainingDataPng_segmentations", help="path to config file",
        type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)

    settings.initialize_cuda_and_logging(cfg)

    inference(cfg)