import numpy as np
import argparse
import os
from tqdm import tqdm
from PIL import Image
import json

subsets = ['train', 'val','test']
subset_folders = ['image',  'mask']

val_pairs = ['1123',
             '1116',
             'pairs_34',
             'pairs_18_20',
             'c_4',
             'ant-B',
             'ant-D',
             'ant2',
             'ant',
             '2507 5 2BC5 ZA3 mag OA20',
             '2019-12-02_TiAl_box3n15_zoom_1_2',
             '2019-12-02_TiAl_box3n15_zoom_4_3',
             '2019-12-02_TiAl_box3n15_zoom_7_5',
             '2019-12-02_TiAl_box3n15_zoom_7_6']

def seg_generator(all_dislocation_points, IMG_H, IMG_W):

    gen_image = np.zeros((IMG_H, IMG_W), np.uint8)
    for disloc in all_dislocation_points:
            for i in range(0,2):
                for j in range(0,2):
                    try:
                        gen_image[int(disloc[1]) + i, int(disloc[0]) - j] = 255
                        gen_image[int(disloc[1]) - i, int(disloc[0]) + j] = 255
                        gen_image[int(disloc[1]) + i, int(disloc[0]) + j] = 255
                        gen_image[int(disloc[1]) - i, int(disloc[0]) - j] = 255

                    except Exception as ex:
                        continue
    return gen_image


def do_intermediate_keypoints(p1, p2, n_points=20):

    x_spacing, y_spacing = (p2[0] - p1[0]) / (n_points + 1), (p2[1] - p1[1]) / (n_points + 1)
    return [[p1[0] + i * x_spacing, p1[1] + i * y_spacing]  for i in range(1, n_points + 1)]


def resize_keypoint(kp1, old_w, old_h, new_w, new_h):

    kp_resized = kp1.copy()
    kp_resized[0], kp_resized[1] = (kp_resized[0] / old_w) * new_w, (kp_resized[1] / old_h) * new_h

    return kp_resized

def generate_extended_image_keypoints(path_to_json_b,
                                      width_b ,height_b,
                                      IMG_W, IMG_H,
                                      intermediate_keypoints):

    with open(path_to_json_b) as f: img_b = json.load(f)
    all_disl_points = []

    for idx, shapes in enumerate(img_b['shapes']):
        shape_b = img_b['shapes'][idx]['points']

        if 'stacking' in shapes['label'] or  'grain' in shapes['label'] : continue

        all_points_for_dislocation_b = []
        count = 0
        for indx in range(1, len(shape_b)):
            p1_b, p2_b = shape_b[count].copy(), shape_b[indx].copy()

            p1_b, p2_b = resize_keypoint(p1_b.copy(), width_b ,height_b, IMG_W, IMG_H), \
                         resize_keypoint(p2_b.copy(), width_b ,height_b, IMG_W, IMG_H )

            count += 1
            keypoints_for_line_b = do_intermediate_keypoints(p1_b, p2_b, intermediate_keypoints)

            all_points_for_dislocation_b.extend(keypoints_for_line_b)

        all_points_for_dislocation_b = np.array(all_points_for_dislocation_b)
        all_disl_points.extend(all_points_for_dislocation_b)

    return all_disl_points

def create_dataset_folders(save_resized_dataset_path):

    for subset in subsets:
        for folder in subset_folders:
            sub_path = os.path.join(save_resized_dataset_path, subset, folder)
            if not os.path.exists(sub_path):  os.makedirs(sub_path)

def create_segmentation_masks(path_to_jsons, path_to_images, subset, args):

    # read json files from folder
    jsons =  [x for x in os.listdir(path_to_jsons) if '.json' in x]
    # read images from folder
    image_paths = [x.replace(path_to_images, path_to_jsons).replace('json','png') for x in jsons]
    # open as image
    images = [Image.open(os.path.join(path_to_images, image_path)) for image_path in image_paths]
    # get image height and width
    image_sizes = [image.size for image in images]
    # resize images to specified width and height
    resized_images = [image.resize((args.IMG_W, args.IMG_H)) for image in images]
    table = [i / 256 for i in range(65536)]
    resized_images = [img.point(table, 'L') if np.array(img).dtype == np.int32 else img for img in resized_images]

    # generate_extended_keypoints
    extended_keypoints = [generate_extended_image_keypoints(os.path.join(path_to_jsons, path_to_json),
                                      image_sizes[idx][0] ,image_sizes[idx][1],
                                      args.IMG_W, args.IMG_H,
                                      args.intermediate_keypoints) for idx, path_to_json in enumerate(jsons)]

    # generate segmentation masks from extended keypoints
    segmentation_masks = [seg_generator(extended_keypoint, args.IMG_W, args.IMG_H) for extended_keypoint in extended_keypoints]

    #save generated images
    for idx, segmenation_mask in enumerate(segmentation_masks):

        segmentation_path = os.path.join(args.path_to_save_images, subset, 'mask', os.path.basename(image_paths[idx]))
        image_path = os.path.join(args.path_to_save_images, subset, 'image', os.path.basename(image_paths[idx]))

        Image.fromarray(segmenation_mask).convert('L').save(segmentation_path)
        resized_images[idx].convert('L').save(image_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Dislocation Segmentation training")

    parser.add_argument('--path_to_save_images', type=str,default='/cvlabdata1/cvlab/datasets_anastasiia/dislocations/segmentation/', help='Path to save datasets')
    parser.add_argument('--IMG_W', type=int, default=512, help='image width')
    parser.add_argument('--IMG_H', type=int, default=512, help='image height')
    parser.add_argument('--intermediate_keypoints', type=int, default=300, help='amount of intermediate keypoints')

    args = parser.parse_args()
    create_dataset_folders(args.path_to_save_images)

    images_paths = ['/cvlabsrc1/cvlab/datasets_anastasiia/dislocations/01_12_21_for_video',
                    '/cvlabsrc1/cvlab/datasets_anastasiia/dislocations/ALL_DATA_fixed_bottom_img_with_semantics']

    for image_folder_path in images_paths:

        folders = os.listdir(image_folder_path)
        for folder in tqdm(folders):

            if '.DS_Store'  in folder:  continue
            path_images = os.path.join(image_folder_path, folder)
            path_to_jsons = os.path.join(image_folder_path, folder, 'results')

            if 'New_dataset_01_05_2020' in folder:
                subset = 'val'
            else:
                subset = 'train'

            create_segmentation_masks(path_to_jsons, path_images, subset, args)