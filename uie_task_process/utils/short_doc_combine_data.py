import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import base64
import copy
import json
import math
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import requests
import yaml
from sklearn.model_selection import train_test_split

IP_ADDRESS = '192.168.106.131'
PORT = 8506


def convert_b64(file):
    if os.path.isfile(file):
        with open(file, 'rb') as fh:
            x = base64.b64encode(fh.read())
            return x.decode('ascii').replace('\n', '')
    else:
        return None


def general(data, ip_address, port):
    r = requests.post(f'http://{ip_address}:{port}/lab/ocr/predict/general', json=data)
    # print(r)
    return r.json()


def get_ocr_results(image_file, ip_address=IP_ADDRESS, port=PORT):
    """
    结构化OCR全文识别结果配置
    """
    data = {
        'scene': 'chinese_print',
        'image': convert_b64(image_file),
        'parameters': {
            'rotateupright': True,
            'refine_boxes': True,
            'sort_filter_boxes': True,
            'support_long_rotate_dense': False,
            'vis_flag': False,
            'sdk': True,
            'det': 'mrcnn-v5.1',
            'recog': 'transformer-v2.8-gamma-faster',
        },
    }

    ret = general(data, ip_address, port)

    return ret['data']['json']['general_ocr_res']


def load_yaml(conf_file):
    """
    :param conf_file: can be file path, or string, or bytes
    :return:
    """
    if os.path.isfile(conf_file):
        return yaml.load(open(conf_file), Loader=yaml.FullLoader)
    else:
        return yaml.load(conf_file, Loader=yaml.FullLoader)


def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)


def list_image(directory, ext='jpg|jpeg|bmp|png|tif|tiff|JPG|PNG|TIF|TIFF'):
    listOfFiles = list()
    for dirpath, dirnames, filenames in os.walk(directory):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    pattern = ext + r'\Z'
    res = [f for f in listOfFiles if re.findall(pattern, f)]
    return res


def rotate_box(box, image_size, angle):
    assert box.shape == (4, 2)
    w, h = image_size
    box_copy = copy.deepcopy(box)
    if angle == 0:
        return box
    if angle == -90:
        box[:, 0] = w - 1 - box_copy[:, 1]
        box[:, 1] = box_copy[:, 0]
        return box
    if angle == 90:
        box[:, 0] = box_copy[:, 1]
        box[:, 1] = h - 1 - box_copy[:, 0]
        return box
    if angle == 180:
        box[:, 0] = w - 1 - box_copy[:, 0]
        box[:, 1] = h - 1 - box_copy[:, 1]
        return box


def rotate_image_only(im, angle):
    """
    rotate image in range[-10,10]
    :param polys:
    :param tags:
    :return:
    """

    def rotate(src, angle, scale=1.0):  # 1
        w = src.shape[1]
        h = src.shape[0]
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        rotated_image = cv2.warpAffine(
            src,
            rot_mat,
            (int(math.ceil(nw)), int(math.ceil(nh))),
            flags=cv2.INTER_LANCZOS4,
        )
        return rotated_image

    old_h, old_w, _ = im.shape
    old_center = (old_w / 2, old_h / 2)

    image = rotate(im, angle)
    new_h, new_w, _ = image.shape
    new_center = (new_w / 2, new_h / 2)

    return image, old_center, new_center


def plot_bbox_on_image(
    image, bboxes, color=(255, 0, 0), show_order=False, rectangle=True
):
    for index, box in enumerate(bboxes):
        box = np.array(box)
        if rectangle:
            x1, y1, x2, y2 = (
                min(box[:, 0]),
                min(box[:, 1]),
                max(box[:, 0]),
                max(box[:, 1]),
            )
            box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        first_point = (int(float(box[0, 0])), int(float(box[0, 1])))
        cv2.circle(image, first_point, 4, (0, 0, 255), 2)
        cv2.polylines(
            image,
            [box.astype(np.int32).reshape((-1, 1, 2))],
            True,
            color=color,
            thickness=2,
        )
        if show_order:
            cv2.putText(
                image,
                str(index),
                (int(float(box[0, 0])), int(float(box[0, 1]))),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                thickness=1,
                color=(0, 0, 255),
            )


def vis_image(dataset_folder):
    base_name = os.path.basename(dataset_folder)
    dir_name = os.path.dirname(dataset_folder)
    save_folder = os.path.join(dir_name, base_name + '_show')
    check_folder(save_folder)

    label_or_ocr_not_exist = []
    image_files = list_image(
        os.path.join(dataset_folder, 'Images', 'train')
    ) + list_image(os.path.join(dataset_folder, 'Images', 'val'))

    for image_file in image_files:
        image = cv2.imread(image_file)
        image_name = os.path.basename(image_file)
        train_or_val = os.path.basename(os.path.dirname(image_file))

        json_name = os.path.splitext(image_name)[0] + '.json'
        json_file = os.path.join(
            dataset_folder, 'Images', 'ocr_results', train_or_val, json_name
        )
        label_file = os.path.join(dataset_folder, 'Labels', json_name)

        if (not os.path.exists(label_file)) or (not os.path.exists(json_file)):
            label_or_ocr_not_exist.append(image_file)
            continue

        with open(json_file, 'r') as f:
            ocr_results = json.load(f)
            rotate_angle = ocr_results['rotate_angle']
            rotateupright = ocr_results['rotateupright']
            text_direction = ocr_results['text_direction']
            bboxes = ocr_results['bboxes']
            texts = ocr_results['texts']
            image_size = ocr_results['image_size']

        with open(label_file, 'r') as f:
            labels = json.load(f)
            categorys = []
            values = []
            gt_bboxes = []
            for label in labels:
                category = label['category']
                value = label['value']
                points = label['points']
                categorys.append(category)
                values.append(value)
                gt_bboxes.append(points)

        for index, bbox in enumerate(gt_bboxes):
            bbox = np.array(bbox)
            gt_bboxes[index] = rotate_box(bbox, image_size, rotate_angle)

        image, _, _ = rotate_image_only(image, rotate_angle)
        plot_bbox_on_image(image, bboxes, show_order=True, rectangle=False)
        plot_bbox_on_image(image, gt_bboxes, color=(0, 0, 255), rectangle=False)
        cv2.imwrite(os.path.join(save_folder, image_name), image)


def convert_label(dataset_folder):
    """
    convert socr format to uie trainval format
    """
    # meta_file = os.path.join(dataset_folder, 'meta.yaml')
    # meta_info = load_yaml(meta_file)
    # field_def = meta_info['field_def']

    base_name = os.path.basename(dataset_folder)
    dir_name = os.path.dirname(dataset_folder)
    save_folder = os.path.join(dir_name, base_name + '_convert')
    save_image_folder = os.path.join(save_folder, 'images')
    save_ocr_folder = os.path.join(save_folder, 'ocr_results')
    save_label_folder = os.path.join(save_folder, 'Labels')
    save_val_image_folder = os.path.join(save_folder, 'val_images')
    check_folder(save_folder)
    check_folder(save_image_folder)
    check_folder(save_val_image_folder)
    check_folder(save_ocr_folder)
    check_folder(save_label_folder)

    label_train_studio = []
    label_val_studio = []
    label_or_ocr_not_exist = []
    image_files = Path(dataset_folder).glob('*/')
    for image_file in image_files:
        image = cv2.imread(image_file)
        image_name = os.path.basename(image_file)
        train_or_val = os.path.basename(os.path.dirname(image_file))

        json_name = os.path.splitext(image_name)[0] + '.json'
        json_file = os.path.join(
            dataset_folder, 'Images', 'ocr_results', train_or_val, json_name
        )
        label_file = os.path.join(dataset_folder, 'Labels', json_name)

        if (not os.path.exists(json_file)) or (not os.path.exists(label_file)):
            label_or_ocr_not_exist.append(image_file)
            continue

        with open(json_file, 'r') as f:
            ocr_results = json.load(f)
            rotate_angle = ocr_results['rotate_angle']
            rotateupright = ocr_results['rotateupright']
            text_direction = ocr_results['text_direction']
            bboxes = ocr_results['bboxes']
            texts = ocr_results['texts']
            image_size = ocr_results['image_size']

        with open(label_file, 'r') as f:
            labels = json.load(f)
            categorys = []
            values = []
            gt_bboxes = []
            for label in labels:
                category = label['category']
                value = label['value']
                points = label['points']
                categorys.append(category)
                values.append(value)
                gt_bboxes.append(points)

        shutil.copy(json_file, os.path.join(save_ocr_folder, json_name))
        shutil.copy(label_file, os.path.join(save_label_folder, json_name))

        # rotate gt_boxes
        for index, bbox in enumerate(gt_bboxes):
            bbox = np.array(bbox)
            gt_bboxes[index] = rotate_box(bbox, image_size, rotate_angle)

        # sort gt_boxes, from top to bottom, from left to right
        sorted_res = sorted(
            enumerate(gt_bboxes), key=lambda x: (x[1][0][1], x[1][0][0])
        )
        gt_bboxes = [elem[1] for elem in sorted_res]
        categorys = [categorys[elem[0]] for elem in sorted_res]
        values = [values[elem[0]] for elem in sorted_res]

        # rotate image
        image, _, _ = rotate_image_only(image, rotate_angle)
        cv2.imwrite(os.path.join(save_image_folder, image_name), image)
        if train_or_val == 'val':
            cv2.imwrite(os.path.join(save_val_image_folder, image_name), image)

        original_height, original_width, _ = image.shape
        convert_label = {}
        convert_label['annotations'] = [dict()]
        result = []
        for index, bbox in enumerate(gt_bboxes):
            # 处理成水平box
            bbox = np.array(bbox)
            x1, y1, x2, y2 = (
                min(bbox[:, 0]),
                min(bbox[:, 1]),
                max(bbox[:, 0]),
                max(bbox[:, 1]),
            )
            info = dict()
            info['original_width'] = int(original_width)
            info['original_height'] = int(original_height)
            info['value'] = dict()
            # if categorys[index] not in field_def:
            #     print(image_file, f'{categorys[index]} does not exist in field_def.')
            #     continue
            info['value']['rectanglelabels'] = [categorys[index]]
            info['value']['x'] = float(x1 / original_width * 100)
            info['value']['y'] = float(y1 / original_height * 100)
            info['value']['width'] = float((x2 - x1) / original_width * 100)
            info['value']['height'] = float((y2 - y1) / original_height * 100)
            info['id'] = os.path.splitext(image_name)[0] + '_' + str(index)
            info['type'] = 'rectanglelabels'
            info['origin_bbox'] = bbox.tolist()
            info['origin_text'] = values[index]
            result.append(info)
        convert_label['annotations'][0]['result'] = result
        convert_label['data'] = {'image': 'prefix-' + image_name}

        if train_or_val == 'train':
            label_train_studio.append(convert_label)
        else:
            label_val_studio.append(convert_label)

    with open(os.path.join(save_folder, 'label_train_studio.json'), 'w') as f:
        f.write(json.dumps(label_train_studio, ensure_ascii=False))
    with open(os.path.join(save_folder, 'label_val_studio.json'), 'w') as f:
        f.write(json.dumps(label_val_studio, ensure_ascii=False))

    print(
        'total image: {}, label_or_ocr_not_exist image: {}'.format(
            len(image_files), len(label_or_ocr_not_exist)
        )
    )


def combine_label(dataset_folder, oup_path):
    """
    combine all scene datasets
    """
    
    dataset_dict = {'train': [], 'val': []}
    label_train_studio = list()
    label_val_studio = list()
    all_labels_oup_path = Path(oup_path) / 'Labels'
    train_images_oup_path = Path(oup_path) / 'Images'
    val_images_oup_path = Path(oup_path) / 'val_images'
    ocr_results_oup_path = Path(oup_path) / 'ocr_results'
    check_folder(train_images_oup_path)
    check_folder(all_labels_oup_path)
    check_folder(val_images_oup_path)
    check_folder(ocr_results_oup_path)
    
    scene_list = list(Path(dataset_folder).glob('[!.]*'))
    for scene in scene_list:
        cur_image_files = list(scene.glob('Images/[!.]*'))
        cur_label_files = list(scene.glob('Labels/[!.]*'))

        assert len(cur_image_files) == len(cur_label_files)
        
        train_imgs, val_imgs = train_test_split(
            cur_image_files,
            train_size=0.8,
            test_size=0.2,
            shuffle=True,
            random_state=42,
        )
        dataset_dict['train'].extend(train_imgs)
        dataset_dict['val'].extend(val_imgs)

        for label in cur_label_files:
            shutil.copy(label, all_labels_oup_path)

        # convert to label studio
        for train_or_val, imgs in dataset_dict.items():
            # todo: 加入多线程
            for img in tqdm(imgs, desc=f'{train_or_val}_{scene.name}'):
                json_name = img.with_suffix('.json').name

                ocr_result = get_ocr_results(str(img))
                rotate_angle = ocr_result['rotate_angle']
                image_size = ocr_result['image_size']
                ocr_json_file_oup = ocr_results_oup_path / json_name
                with open(ocr_json_file_oup, 'w') as f:
                    json.dump(ocr_result, f, ensure_ascii=False, indent=2)

                label_file = all_labels_oup_path / json_name
                with open(label_file, 'r') as f:
                    labels = json.load(f)
                    categorys = []
                    values = []
                    gt_bboxes = []
                    for label in labels:
                        category = label['category']
                        value = label['value']
                        points = label['points']
                        categorys.append(category)
                        values.append(value)
                        gt_bboxes.append(points)

                # rotate gt_boxes
                for index, bbox in enumerate(gt_bboxes):
                    bbox = np.array(bbox)
                    gt_bboxes[index] = rotate_box(bbox, image_size, rotate_angle)

                # sort gt_boxes, from top to bottom, from left to right
                sorted_res = sorted(
                    enumerate(gt_bboxes), key=lambda x: (x[1][0][1], x[1][0][0])
                )
                gt_bboxes = [elem[1] for elem in sorted_res]
                categorys = [categorys[elem[0]] for elem in sorted_res]
                values = [values[elem[0]] for elem in sorted_res]

                # rotate image
                image = cv2.imread(str(img))
                image_name = img.name
                image, _, _ = rotate_image_only(image, rotate_angle)
                cv2.imwrite(os.path.join(train_images_oup_path, image_name), image)
                if train_or_val == 'val':
                    cv2.imwrite(os.path.join(val_images_oup_path, image_name), image)

                original_height, original_width, _ = image.shape
                convert_label = {}
                convert_label['annotations'] = [dict()]
                result = []
                for index, bbox in enumerate(gt_bboxes):
                    # 处理成水平box
                    bbox = np.array(bbox)
                    x1, y1, x2, y2 = (
                        min(bbox[:, 0]),
                        min(bbox[:, 1]),
                        max(bbox[:, 0]),
                        max(bbox[:, 1]),
                    )
                    info = dict()
                    info['original_width'] = int(original_width)
                    info['original_height'] = int(original_height)
                    info['value'] = dict()
                    info['value']['rectanglelabels'] = [categorys[index]]
                    info['value']['x'] = float(x1 / original_width * 100)
                    info['value']['y'] = float(y1 / original_height * 100)
                    info['value']['width'] = float((x2 - x1) / original_width * 100)
                    info['value']['height'] = float((y2 - y1) / original_height * 100)
                    info['id'] = os.path.splitext(image_name)[0] + '_' + str(index)
                    info['type'] = 'rectanglelabels'
                    info['origin_bbox'] = bbox.tolist()
                    info['origin_text'] = values[index]
                    result.append(info)
                convert_label['annotations'][0]['result'] = result
                convert_label['data'] = {
                    'image': 'prefix-' + image_name,
                    'scene': f'{scene.name}',
                }

                if train_or_val == 'train':
                    label_train_studio.append(convert_label)
                else:
                    label_val_studio.append(convert_label)

    with open(Path(oup_path) / 'label_train_studio.json', 'w') as f:
        f.write(json.dumps(label_train_studio, ensure_ascii=False))
    with open(Path(oup_path) / 'label_val_studio.json', 'w') as f:
        f.write(json.dumps(label_val_studio, ensure_ascii=False))


if __name__ == '__main__':
    # dataset_folder = '/Users/gulixin/Desktop/数据/卡证表单智能结构化数据/第一批标注数据/v1/增值税普票'
    # vis_image(dataset_folder)

    dataset_folder = '/mnt/disk0/youjiachen/label_studio/output'
    dst = '/mnt/disk0/youjiachen/label_studio/output_combine'

    combine_label(dataset_folder, dst)
