import json
import numpy as np
from PIL import Image

def do_intermediate_keypoints(p1, p2, n_points=20):

    x_spacing, y_spacing = (p2[0] - p1[0]) / (n_points + 1), (p2[1] - p1[1]) / (n_points + 1)
    return [[p1[0] + i * x_spacing, p1[1] + i * y_spacing]  for i in range(1, n_points + 1)]

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

def generate_extended_image_keypoints(path_to_json_b, intermediate_keypoints):

    with open(path_to_json_b) as f: img_b = json.load(f)
    all_disl_points = []

    for idx, shapes in enumerate(img_b['shapes']):
        shape_b = img_b['shapes'][idx]['points']

        if 'stacking' in shapes['label'] or  'grain' in shapes['label'] : continue

        all_points_for_dislocation_b = []
        count = 0
        for indx in range(1, len(shape_b)):
            p1_b, p2_b = shape_b[count].copy(), shape_b[indx].copy()

            count += 1
            keypoints_for_line_b = do_intermediate_keypoints(p1_b, p2_b, intermediate_keypoints)

            all_points_for_dislocation_b.extend(keypoints_for_line_b)

        all_points_for_dislocation_b = np.array(all_points_for_dislocation_b)
        all_disl_points.extend(all_points_for_dislocation_b)

    return all_disl_points


if __name__ == '__main__':

    json_path = '/Users/anastasiia.mishchuk/Downloads/Dataset2_31072020/35 2BC6 CL=330-30-1.json'
    segmentation_path = json_path.replace('.json','_gt.png')
    intermediate_keypoints = 100

    extended_keypoints = generate_extended_image_keypoints(json_path, intermediate_keypoints)
    segmentation_mask = seg_generator(extended_keypoints, 1024,1024)
    Image.fromarray(segmentation_mask).convert('L').save(segmentation_path)
