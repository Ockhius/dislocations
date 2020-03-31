import json
import os
import numpy as np
import cv2
from PIL import Image

subsets = ['train', 'test']

subset_folders = ['disparity',
                  'left_image',
                  'right_image',
                  'segmentation_right_image',
                  'segmentation_left_image']

def create_dataset_folders(save_resized_dataset_path):

    for subset in subsets:
        for folder in subset_folders:
            sub_path = os.path.join(save_resized_dataset_path, subset, folder)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

def do_intermediate_keypoints(p1, p2, n_points=20):

    x_spacing = (p2[0] - p1[0]) / (n_points + 1)
    y_spacing = (p2[1] - p1[1]) / (n_points + 1)

    return [[p1[0] + i * x_spacing, p1[1] + i * y_spacing]
            for i in range(1, n_points + 1)]

def resize_keypoint(kp1, old_w, old_h, new_w, new_h):
    kp1 = kp1.copy()
    kp1[0] = (kp1[0]/old_w)*new_w
    kp1[1] = (kp1[1]/old_h)*new_h

    return kp1

def resize_images(path_to_images, b_json, d_json, IMG_W, IMG_H):

    b_img_path = path_to_images + '/' + os.path.basename(b_json).replace('json', 'png')
    d_img_path = path_to_images + '/' + os.path.basename(d_json).replace('json', 'png')

    b_image, d_image = Image.open(b_img_path), Image.open(d_img_path)

    width_b ,height_b = b_image.size

    b_image = b_image.resize((IMG_W, IMG_H))
    d_image = d_image.resize((IMG_W, IMG_H))

    return b_image, d_image, width_b ,height_b


def generate_path_to_save_image(path_to_save_images, subset, folder, pair, path_to_json_b):
    '''

    :param path_to_save_images:
    :param subset: train or test
    :param folder: name of folder, ex. left, right, disparity, etc.
    :param pair: path to the json file
    :param path_to_json_b:
    :return: name generated in format : save_folder/train/left/image_name_LEFT.png
    '''

    if 'right' in folder:

        name = os.path.join(path_to_save_images,
                            subset,
                            folder,
                            pair + '_' +
                            os.path.basename(path_to_json_b).replace('json', 'png').
                                                             replace('.png','_RIGHT.png'))
    else:
         name = os.path.join(path_to_save_images,
                            subset,
                            folder,
                            pair + '_' +
                            os.path.basename(path_to_json_b).replace('json', 'png').
                                                             replace('.png', '_LEFT.png'))
    return name.replace('\'','')


def seg_generator(all_dislocation_points, gen_images, indx):

    for disloc in all_dislocation_points:
            for i in range(0,2):
                for j in range(0,2):
                    try:
                        gen_images[indx][int(disloc[1]) + i, int(disloc[0]) - j] = 255
                        gen_images[indx][int(disloc[1]) - i, int(disloc[0]) + j] = 255
                        gen_images[indx][int(disloc[1]) + i, int(disloc[0]) + j] = 255
                        gen_images[indx][int(disloc[1]) - i, int(disloc[0]) - j] = 255

                    except Exception as ex:
                        continue
    return gen_images

def disp_generator(all_dislocation_points, gen_images, disp, indx):

    for idx, disloc in enumerate(all_dislocation_points):

            for i in range(0,2):
                for j in range(0,2):
                    try:
                        gen_images[indx][int(disloc[1]) + i, int(disloc[0]) - j]  = -disp[idx][0].round() + 127
                        gen_images[indx][int(disloc[1]) - i, int(disloc[0]) + j]  = -disp[idx][0].round() + 127
                        gen_images[indx][int(disloc[1]) + i, int(disloc[0]) + j]  = -disp[idx][0].round() + 127
                        gen_images[indx][int(disloc[1]) - i, int(disloc[0]) - j]  = -disp[idx][0].round() + 127
                    except Exception as ex:
                        continue
    return gen_images


def generate_segmentation_images(all_disloc_points_img_b, all_disloc_points_img_d):

    generated_segmentation = [np.zeros((IMG_H, IMG_W,1), np.uint8) for i in range(0,2)]

    seg_generator(all_disloc_points_img_b, generated_segmentation, 0)
    seg_generator(all_disloc_points_img_d, generated_segmentation, 1)

    return generated_segmentation

def generate_disparity_images(all_disloc_points_img_b, all_disloc_points_img_d):

    generated_disparity = [np.zeros((IMG_H, IMG_W,1), np.uint8) for i in range(0,2)]

    disp_l = np.array(all_disloc_points_img_b) - np.array(all_disloc_points_img_d)
    disp_r = np.array(all_disloc_points_img_d) - np.array(all_disloc_points_img_b)

    disp_generator(all_disloc_points_img_b, generated_disparity, disp_l, 0)
    disp_generator(all_disloc_points_img_d, generated_disparity, disp_r, 1)

    return generated_disparity

def generate_polygons(path_to_json_b, path_to_json_d,img_b_segmentation, img_d_segmentation,width_b ,height_b):

    with open(path_to_json_b) as f: img_b = json.load(f)
    with open(path_to_json_d) as f: img_d = json.load(f)

    img_b_, img_d_ = img_b_segmentation.squeeze(), img_d_segmentation.squeeze()

    for idx, shapes in enumerate(img_b['shapes']):

        if 'stacking' in shapes['label'] or  'grain' in shapes['label'] :

            shape_b, shape_d = img_b['shapes'][idx]['points'],  img_d['shapes'][idx]['points']

            p1_b, p2_b = shape_b.copy(), shape_d.copy()
            p1_resized, p2_resized = [], []

            for i in range(0, len(p1_b)):

                p1_r = resize_keypoint(np.array(p1_b)[i], width_b ,height_b, IMG_W, IMG_H)
                p2_r = resize_keypoint(np.array(p2_b)[i], width_b ,height_b, IMG_W, IMG_H)

                p1_resized.append(p1_r)
                p2_resized.append(p2_r)

            p1_b, p2_b = np.array(p1_resized) , np.array(p2_resized)

            cv2.fillPoly(img_b_, [np.array([[x, y] for x, y in p1_b], 'int32')],  125)
            cv2.fillPoly(img_d_, [np.array([[x, y] for x, y in p2_b], 'int32')],  125)

    return img_b_, img_d_

def generate_extended_image_keypoints(path_to_json_b, path_to_json_d,
                                      width_b ,height_b, intermediate_keypoints):

    with open(path_to_json_b) as f: img_b = json.load(f)
    with open(path_to_json_d) as f: img_d = json.load(f)

    all_dislocation_points_img_b, all_dislocation_points_img_d = [], []

    for idx, shapes in enumerate(img_b['shapes']):

        shape_b, shape_d = img_b['shapes'][idx]['points'], img_d['shapes'][idx]['points']

        if 'stacking' in shapes['label'] or \
                'grain' in shapes['label'] :
            continue

        all_points_for_dislocation_b, all_points_for_dislocation_d = [], []
        count = 0
        for indx in range(1, len(shape_b)):
            p1_b, p2_b = shape_b[count].copy(), shape_b[indx].copy()

            p1_b, p2_b = resize_keypoint(p1_b.copy(), width_b ,height_b, IMG_W, IMG_H), \
                         resize_keypoint(p2_b.copy(), width_b ,height_b, IMG_W, IMG_H )

            count += 1
            keypoints_for_line_b = do_intermediate_keypoints(p1_b, p2_b, intermediate_keypoints)

            all_points_for_dislocation_b.extend(keypoints_for_line_b)

        all_points_for_dislocation_b = np.array(all_points_for_dislocation_b)
        count = 0
        for indx in range(1, len(shape_d)):
            p1_d, p2_d = shape_d[count], shape_d[indx]

            p1_d, p2_d  = resize_keypoint(p1_d, width_b ,height_b, IMG_W, IMG_H), \
                         resize_keypoint(p2_d, width_b ,height_b, IMG_W, IMG_H )

            count += 1
            keypoints_for_line_d = do_intermediate_keypoints(p1_d, p2_d, intermediate_keypoints)

            all_points_for_dislocation_d.extend(keypoints_for_line_d)

        all_points_for_dislocation_b, all_points_for_dislocation_d = np.array(all_points_for_dislocation_b), np.array(all_points_for_dislocation_d)
        all_dislocation_points_img_b.extend(all_points_for_dislocation_b), all_dislocation_points_img_d.extend(all_points_for_dislocation_d)

    return all_dislocation_points_img_b, all_dislocation_points_img_d


def create_disparity_and_segmentation_images(subset, path_to_images, save_path, path_to_jsons):

    jsons = [x for x in os.listdir(path_to_jsons) if '.json' in x]

    if '-B' not in jsons[0]:
        jsons = list(reversed(jsons))

    path_to_json_b = os.path.join(path_to_jsons, jsons[0])
    path_to_json_d = os.path.join(path_to_jsons, jsons[1])

    # resize images to the specified height, width
    b_image, d_image, width_b ,height_b = resize_images(path_to_images, path_to_json_b, path_to_json_d, IMG_W, IMG_H)

    pair = path_to_images.split('/')[-1]

    check = True
    for pair_ in test_pairs:
        if pair_ in path_to_json_d:
            subset = 'test'
            check = False
    if check:
        subset = 'train'


    left_name, right_name = generate_path_to_save_image(save_path, subset, 'left_image', pair, path_to_json_b), \
                            generate_path_to_save_image(save_path, subset, 'right_image', pair, path_to_json_b)

    b_image.save(left_name), d_image.save(right_name)

    all_dislocation_points_img_b, all_dislocation_points_img_d = generate_extended_image_keypoints(path_to_json_b, path_to_json_d,
                                                                                                   width_b ,height_b, 300)
    img_b_segmentation, img_d_segmentation = generate_segmentation_images(all_dislocation_points_img_b, all_dislocation_points_img_d)

    # given preprocessed keypoints, generate images

    all_dislocation_points_img_b, all_dislocation_points_img_d = generate_extended_image_keypoints(path_to_json_b, path_to_json_d,
                                                                                                   width_b ,height_b, intermediate_keypoints)

    disparity_img, disparity_img_r = generate_disparity_images(all_dislocation_points_img_b, all_dislocation_points_img_d)

    img_b_segmentation, img_d_segmentation = generate_polygons(path_to_json_b, path_to_json_d,img_b_segmentation, img_d_segmentation, width_b ,height_b)
    left_segmentation_name, \
    right_segmentation_name, \
    disparity_left = generate_path_to_save_image(save_path, subset, 'segmentation_left_image', pair, path_to_json_b), \
                     generate_path_to_save_image(save_path, subset, 'segmentation_right_image', pair, path_to_json_b), \
                     generate_path_to_save_image(save_path, subset, 'disparity', pair, path_to_json_b),


    cv2.imwrite(left_segmentation_name, img_b_segmentation)
    cv2.imwrite(right_segmentation_name, img_d_segmentation)
    cv2.imwrite(disparity_left, disparity_img)

    print('Processed: {}'.format(left_segmentation_name))

if __name__ == '__main__':

    path_to_images = '/cvlabsrc1/cvlab/datasets_anastasiia/dislocations/ALL_DATA_fixed_bottom_img_with_semantics/'
    path_to_save_images = '/cvlabsrc1/cvlab/datasets_anastasiia/dislocations/ALL_DATA_fixed_bottom_img_with_semantics_resized/'

    subset = 'train'

    IMG_W, IMG_H = 512, 512
    intermediate_keypoints = 300

    test_pairs = ['1123', '1116', 'pairs_34', 'pairs_18_20',  'ant-B',
                  'ant-D', 'ant2',  '2507 5 2BC5 ZA3 mag OA20',
                  'test1_semantics',  'test2_semantics',  'test3_semantics',
                  'test4', 'test5','test6',
                  '2019-12-02_TiAl_box3n15_zoom_1_2',
                  '2019-12-02_TiAl_box3n15_zoom_4_3',
                  '2019-12-02_TiAl_box3n15_zoom_7_5',
                  '2019-12-02_TiAl_box3n15_zoom_7_6',
                  'TiAl_box3n15_pairs_0_2',
                  'TiAl_box3n15_pairs_0_4',
                  'TiAl_box3n15_pairs_0_7',
                  'TiAl_box3n15_pairs_0_16',
                  'TiAl_box3n15_pairs_n2_2',
                  'TiAl_box3n15_pairs_n4_4']

    # create dataset structure of folders
    create_dataset_folders(path_to_save_images)

    # iterate over folders to save resized dataset in the proper structure
    folders = os.listdir(path_to_images)

    for folder in folders:
        if '.DS_Store'  in folder:  continue
        path_images = os.path.join(path_to_images, folder)
        path_to_jsons = os.path.join(path_to_images, folder, 'results')

        create_disparity_and_segmentation_images(subset, path_images, path_to_save_images, path_to_jsons)