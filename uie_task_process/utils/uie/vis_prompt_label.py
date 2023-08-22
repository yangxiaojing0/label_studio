import argparse
import copy
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)


def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = img.astype(np.uint8)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        os.path.join(os.path.dirname(__file__), './simsun.ttc'),
        textSize,
        encoding="utf-8",
    )
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


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


def vis_prompt_label(data_folder, prompt_label_file, res_folder=''):
    save_folder = os.path.join(data_folder, 'vis_prompt_label')
    check_folder(save_folder)

    with open(prompt_label_file, 'r', encoding='utf-8') as f:
        contents = []
        prompts = []
        image_files = []
        result_lists = []
        for line in f.readlines():
            label = json.loads(line)# 第一个gt标签信息
            content = label['content']# 整页文本识别内容
            result_list = label['result_list']
            if len(result_list) == 0:
                continue
            prompt = label['prompt']
            image_file = label['image_file']
            if image_file not in image_files:
                contents.append(content)
                prompts.append([prompt])
                result_lists.append([result_list])
                image_files.append(image_file)
            else:
                index = image_files.index(image_file)
                prompts[index].append(prompt)
                result_lists[index].append(result_list)

    for index, image_file in enumerate(tqdm(image_files)):
        image = cv2.imread(os.path.join(data_folder, 'images', image_file))
        h, w, c = image.shape
        json_name = os.path.splitext(image_file)[0] + '.json'

        with open(os.path.join(data_folder, 'ocr_results', json_name), 'r') as f:
            ocr_results = json.load(f)
            rotate_angle = ocr_results['rotate_angle']
            bboxes = ocr_results['bboxes']
            texts = ocr_results['texts']
            image_size = ocr_results['image_size']

        with open(os.path.join(data_folder, 'Labels', json_name), 'r') as f:
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

        plot_bbox_on_image(image, bboxes, show_order=True, rectangle=False)
        plot_bbox_on_image(image, gt_bboxes, color=(0, 0, 255), rectangle=False)

        image = np.concatenate((image, np.ones((h, w, c)) * 255), axis=1)
        img_prompts = prompts[index]
        img_result_lists = result_lists[index]
        delta = h // (len(img_prompts) * 2)
        for idx, prompt in enumerate(img_prompts):
            point = (int(w), int((2 * idx) * delta))
            image = cv2AddChineseText(image, prompt, point, (255, 0, 0), 30)
            res = img_result_lists[idx]
            res = sorted(res, key=lambda x: x['start'])
            combine_value = ''
            for elem in res:
                combine_value += elem['text'] + '\n'
            point = (int(w), int((2 * idx + 0.5) * delta))
            image = cv2AddChineseText(image, combine_value, point, (0, 0, 255), 30)

        res_file = os.path.join(res_folder, json_name)
        if os.path.exists(res_file):
            image = np.concatenate((image, np.ones((h, w, c)) * 255), axis=1)
            with open(res_file, 'r') as f:
                results = json.load(f)['results']
            if not len(results.keys()):
                continue
            delta = h // (len(results.keys()) * 2)

            offest = 0
            for idx, prompt in enumerate(img_prompts):
                if prompt not in results.keys():
                    continue
                point = (int(2 * w), int((2 * offest) * delta))
                image = cv2AddChineseText(image, prompt, point, (255, 0, 0), 30)
                res = results.pop(prompt)
                combine_value = ''
                for elem in res:
                    combine_value += elem + '\n'
                point = (int(2 * w), int((2 * offest + 0.5) * delta))
                image = cv2AddChineseText(image, combine_value, point, (0, 0, 255), 30)
                offest += 1

            for idx, prompt in enumerate(results.keys()):
                point = (int(2 * w), int((2 * offest) * delta))
                image = cv2AddChineseText(image, prompt, point, (255, 0, 0), 30)
                res = results[prompt]
                combine_value = ''
                for elem in res:
                    combine_value += elem + '\n'
                point = (int(2 * w), int((2 * offest + 0.5) * delta))
                image = cv2AddChineseText(image, combine_value, point, (0, 0, 255), 30)
                offest += 1

        cv2.imwrite(os.path.join(save_folder, image_file), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        default="/home/youjiachen/workspace/relation_exp/任免公告",
        type=str,
        help="",
    )
    parser.add_argument(
        "--prompt_label_file",
        default="/home/youjiachen/workspace/relation_exp/任免公告/dataelem_ocr_v2_ratio_1/dev.txt",
        type=str,
        help="",
    )
    parser.add_argument(
        "--res_folder",
        default="/home/youjiachen/workspace/relation_exp/任免公告/val_res",
        type=str,
        help="",
    )
    args = parser.parse_args()

    # vis_prompt_label(args.data_folder, args.prompt_label_file, args.res_folder)
    
    '''单场景错例分析'''
    data_folder = r'/home/youjiachen/workspace/yxj/uie_task/0817_23scenes_doc/data/0817_23scenes_doc_convert/起诉书'
    prompt_label_file = r'/home/youjiachen/workspace/yxj/uie_task/0817_23scenes_doc/data/0817_23scenes_doc_convert/起诉书/dataelem_ocr_v2/dev.txt'
    res_folder = r'/home/youjiachen/workspace/yxj/uie_task/0817_23scenes_doc/data/0817_23scenes_doc_convert/起诉书/val_res'
    vis_prompt_label(data_folder, prompt_label_file, res_folder)
    
    '''多场景可视化错例分析图片'''
    def draw_img_for_error_analysis(scenes_folder):
        scene_list=list(Path(scenes_folder).glob('[!.]*'))
        for i in scene_list:
            print(str(i)+':')
            data_folder=str(i)
            prompt_label_file=Path(i)/'dataelem_ocr_v2'/'dev.txt'
            res_folder=Path(i)/'val_res'
            vis_prompt_label(data_folder, prompt_label_file, res_folder)
    scenes_folder='/home/youjiachen/workspace/yxj/uie_task/0817_23scenes_doc/data/0817_23scenes_doc_convert'
    # draw_img_for_error_analysis(scenes_folder)
    
    
    
