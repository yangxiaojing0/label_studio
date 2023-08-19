import json
import shutil
from collections import defaultdict
from pathlib import Path
from pprint import pprint

from tqdm import tqdm


def stats_negNpos(data_path):
    s_dict = defaultdict(lambda: {'正例': 0, '负例': 0})
    # s_dict = defaultdict(dict)
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            prompt = data['prompt']
            if not len(data['result_list']):
                s_dict[prompt]['负例'] += 1
            else:
                s_dict[prompt]['正例'] += 1

        pprint(s_dict)


if __name__ == '__main__':
    # img_path = '/mnt/disk0/youjiachen/workspace/contract/施工合同/Images'
    # ocr_path = '/mnt/disk0/youjiachen/workspace/contract/ocr_results'
    # dst = Path(img_path).parent / 'dataelem_ocr_res'

    # img_paths = list(Path(img_path).glob('[!.]*.*'))

    # for img_file in tqdm(img_paths):
    #     json_name = img_file.with_suffix('.json').name
    #     cur_ocr_path = Path(ocr_path) / json_name
    #     shutil.copy(cur_ocr_path, dst)

    val_img = '/mnt/disk0/youjiachen/workspace/ticket-0723_combine/dataelem_ocr_v2/dev_image.txt'
    train_img = '/mnt/disk0/youjiachen/workspace/ticket-0723_combine/dataelem_ocr_v2/train_image.txt'
    
    # with open(val_img, 'r') as f:
    #     print(len(json.load(f)))
    # with open(train_img, 'r') as f:
    #     print(len(json.load(f)))
    
    src = '/mnt/disk0/youjiachen/workspace/ticket-0723'
    scene_list = list(Path(src).glob('[!.]*'))
    for scene in scene_list:
        print(scene.name)