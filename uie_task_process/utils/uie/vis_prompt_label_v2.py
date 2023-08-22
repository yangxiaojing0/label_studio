import argparse
import copy
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

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
        if rectangle:  # 矩形框，按照位置重新排序，我们绘制时，不会重新排序，因为标注和预测是按照阅读顺序标注的
            x1, y1, x2, y2 = (
                min(box[:, 0]),
                min(box[:, 1]),
                max(box[:, 0]),
                max(box[:, 1]),
            )
            # 重新排序？
            box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        first_point = (int(float(box[0, 0])), int(float(box[0, 1])))  # x1, y1
        cv2.circle(image, first_point, 4, (0, 0, 255), 2)  # 绘制圆圈：图像，中心，半径，颜色，线条粗细
        cv2.polylines(
            image,
            [box.astype(np.int32).reshape((-1, 1, 2))],
            True,
            color=color,
            thickness=2,
        )  # 绘制多边形：图像，box，闭合轮廓，颜色，线条粗细
        if show_order:
            cv2.putText(
                image,
                str(index),  # 框索引
                (int(float(box[0, 0])), int(float(box[0, 1]))),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                thickness=1,
                color=(0, 0, 255),
            )  # 绘制圆圈旁边的数字，代表第一个框


def vis_prompt_label(data_folder, prompt_label_file, res_folder=''):
    save_folder = os.path.join(data_folder, 'vis_prompt_label')
    check_folder(save_folder)

    '''读取dev.txt'''
    with open(prompt_label_file, 'r', encoding='utf-8') as f:
        contents = []
        prompts = []
        image_files = []
        result_lists = []
        for line in f.readlines():  # 遍历这个图片的所有标签
            label = json.loads(line)  # 一个gt标签
            content = label['content']  # 图片全文识别内容
            result_list = label['result_list']  # uie的预测的这个标签的结果
            if len(result_list) == 0:  # 这个图片没有标签，直接跳过这个图片
                continue
            prompt = label['prompt']  # 提示词/标注的字段
            image_file = label['image_file']  # 图像数据
            if image_file not in image_files:  # 说明是一张新图像
                contents.append(content)  # 添加该图片识别内容
                prompts.append([prompt])  # 添加字段名
                result_lists.append([result_list])  # 添加该标签预测结果
                image_files.append(image_file)  # 添加图像
            else:
                # 是同一图片的不同框
                # 只需要指明图像是上面创建的图像列表中的第几张，然后添加字段名和预测结果
                index = image_files.index(image_file)
                prompts[index].append(prompt)  # [[第一张图的所有字段名],[第二张图的所有字段名],[]]
                result_lists[index].append(result_list)  # [[{一条结果一个框}，{}一张图片]，[{}]所有图片]

    '''遍历val_img列表, 逐个绘图'''
    for index, image_file in enumerate(tqdm(image_files)):
        image = cv2.imread(
            os.path.join(data_folder, 'images', image_file)
        )  # 从’data_folder/images‘里面找到这张图，读进来
        h, w, c = image.shape
        json_name = os.path.splitext(image_file)[0] + '.json'  # json名称

        '''输入： 找到打开对应的ocr_res 的json文件(蓝色框)'''
        '''得到： 蓝框信息'''
        with open(
            os.path.join(
                data_folder,
                'ocr_results',
                json_name,
            ),
            'r',
            encoding='utf-8',
        ) as f:
            ocr_results = json.load(f)  # ocr_res,蓝框信息
            rotate_angle = ocr_results['rotate_angle']  # 预测的图像的旋转角
            bboxes = ocr_results[
                'bboxes'
            ]  # 所有蓝色框，[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            texts = ocr_results['texts']  # 所有蓝色框的文本信息
            image_size = ocr_results['image_size']  # img的w h

        '''找到打开对应的gt 的json文件(红色框),得到gt结果'''
        # ocr_studio 的 gt的json文件的格式：points，category，value
        with open(
            os.path.join(data_folder, 'Labels', json_name), 'r', encoding='UTF-8'
        ) as f:
            labels = json.load(f)  # gt的json，红框信息
            categorys = []  # 类别
            values = []  # value
            gt_bboxes = []  # points
            for label in labels:
                category = label['category']
                value = label['value']
                points = label['points']
                categorys.append(category)
                values.append(value)
                gt_bboxes.append(points)

        '''在原图上绘制框'''
        # 参数解释：
        # image：原始图像，w*h*3
        # bboxes：所有box，其中box[[498, 127], [527, 127], [527, 155], [498, 155]]
        plot_bbox_on_image(image, bboxes, show_order=True, rectangle=False)  # ocr蓝框
        plot_bbox_on_image(image, gt_bboxes, color=(0, 0, 255), rectangle=False)  # gt红框

        '''绘制label匹配结果ocr_res'''
        # 默认先在横轴上拼接一个一样大小的纯白图
        image = np.concatenate((image, np.ones((h, w, c)) * 255), axis=1)
        img_prompts = prompts[index]  # dev的字段名
        img_result_lists = result_lists[index]  # 字段名对应的识别结果

        # 分配高度决定字号
        delta = h // (len(img_prompts) * 2)  # 一个字段（字段名+text）能分配到的高度
        ocr_word_size = min(30,int(delta / 2))  # note：字号改进

        ocr_contents = defaultdict()# 定义文字内容字典，用于比较不同
        for idx, prompt in enumerate(img_prompts):
            # 平均排布，排序依次写prompt
            # 参数：img, text, position（左上角位置）, textColor=(255, 0, 0) RGB红, textSize
            point = (int(w), int((2 * idx) * delta))

            # note: 不同之处突出显示，Case1: Ocr label有prompt，模型预估没有
            # 预先读取模型预估结果
            res_file = os.path.join(res_folder, json_name)
            if os.path.exists(res_file):
                with open(res_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)['results']  # 字典
            else:
                results = defaultdict()

            if prompt in results.keys():
                image = cv2AddChineseText(
                    image, prompt, point, (255, 0, 0), ocr_word_size
                )  # key红色
            else:
                image = cv2AddChineseText(
                    image, prompt, point, (0, 255, 0), ocr_word_size
                )  # key绿色

            # 平均排序，排序依次写res_text
            res = img_result_lists[idx]
            res = sorted(res, key=lambda x: x['start'])  # 按照start的位置排序
            combine_value = ''
            for elem in res:
                combine_value += elem['text'] + '\n'  # 获得res文本
            point = (
                int(w),
                int((2 * idx + 0.5) * delta),
            )  # 一个字段（字段名+text）能分配到的高度的一半
            image = cv2AddChineseText(
                image, combine_value, point, (0, 0, 255), ocr_word_size
            )  # 蓝色字体
            ocr_contents[prompt] = combine_value

        '''绘制模型预估结果val_res'''
        res_file = os.path.join(res_folder, json_name)
        if os.path.exists(res_file):
            # 在绘制cor_res的基础上再往后拼接一个
            image = np.concatenate((image, np.ones((h, w, c)) * 255), axis=1)
            with open(res_file, 'r', encoding='utf-8') as f:
                results = json.load(f)['results']  # 字典
            if not len(results.keys()):
                continue
            delta = h // (len(results.keys()) * 2)  # 平均排布
            eval_word_size = min(30,int(delta / 2))

            offest = 0
            # 上面先画标注中有的字段
            for idx, prompt in enumerate(img_prompts):
                if prompt not in results.keys():
                    continue
                point = (int(2*w), int((2 * offest) * delta))

                image = cv2AddChineseText(
                    image, prompt, point, (255, 0, 0), eval_word_size
                )  # 红

                res = results.pop(prompt)

                combine_value = ''
                for elem in res:
                    combine_value += elem + '\n'
                point = (int(2 * w), int((2 * offest + 0.5) * delta))

                # note: 不同之处突出显示，Case2: 模型预估的contents和ocr label的contents不一样
                if combine_value in list(ocr_contents.values()):
                    image = cv2AddChineseText(
                        image, combine_value, point, (0, 0, 255), eval_word_size
                    )# value蓝色
                else:
                    image = cv2AddChineseText(
                        image, combine_value, point, (0, 255, 255), eval_word_size
                    ) # value天蓝色
                offest += 1

            # 下面后画只有模型预估有的字段
            # note: 不同之处突出显示，Case3: 模型预估有prompt，ocr label没有
            for idx, prompt in enumerate(results.keys()):
                point = (int(2 * w), int((2 * offest) * delta))
                image = cv2AddChineseText(
                    image, prompt, point, (0, 255, 0), eval_word_size
                )  # key绿色
                res = results[prompt]
                combine_value = ''
                for elem in res:
                    combine_value += elem + '\n'
                point = (int(2 * w), int((2 * offest + 0.5) * delta))
                image = cv2AddChineseText(
                    image, combine_value, point, (0, 255, 255), eval_word_size
                ) # value天蓝色
                offest += 1

        cv2.imwrite(os.path.join(save_folder, image_file), image)  # 储存


if __name__ == '__main__':
    '''单场景错例分析'''
    data_folder = r'F:\desk\debug-vis_val_res\daolu'
    prompt_label_file = r'F:\desk\debug-vis_val_res\daolu\dataelem_ocr_v2/dev.txt'
    res_folder = r'F:\desk\debug-vis_val_res\daolu\\val_res'
    # vis_prompt_label(data_folder, prompt_label_file, res_folder)

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
    draw_img_for_error_analysis(scenes_folder)
