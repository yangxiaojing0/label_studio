import json
from pathlib import Path


def combine_data_to_uie_trainval_txt(src, dst):
    Path(dst).mkdir(exist_ok=True, parents=True)
    train_ds_list = list(Path(src).glob('[!.|混合]*/**/train.txt'))
    train_img_ds_list = list(Path(src).glob('[!.|混合]*/**/train_image.txt'))

    with open(Path(dst) / 'train.txt', 'w') as f1:
        for train_ds in train_ds_list:
            with open(train_ds, 'r') as f2:
                for line in f2:
                    f1.write(line)

    train_img_txt = {}
    with open(Path(dst) / 'train_image.txt', 'w') as f1:
        for train_img_ds in train_img_ds_list:
            with open(train_img_ds, 'r') as f2:
                cur_data = json.load(f2)
                train_img_txt.update(cur_data)
        json.dump(train_img_txt, f1, ensure_ascii=False)

    pre_data = {}
    val_ds_list = list(Path(src).glob('[!.|混合]*/**/dev.txt'))
    val_img_ds_list = list(Path(src).glob('[!.|混合]*/**/dev_image.txt'))
    val_img_txt = {}

    with open(Path(dst) / 'dev.txt', 'w') as f1:
        for val_ds in val_ds_list:
            with open(val_ds, 'r') as f2:
                for line in f2:
                    f1.write(line)

    with open(Path(dst) / 'dev_image.txt', 'w') as f1:
        for val_img_ds in val_img_ds_list:
            with open(val_img_ds, 'r') as f2:
                cur_data = json.load(f2)
                val_img_txt.update(cur_data)
        json.dump(val_img_txt, f1, ensure_ascii=False)


if __name__ == '__main__':
    src = '/mnt/disk0/youjiachen/workspace/short_doc/short_doc_0707_trainval'
    dst = '/mnt/disk0/youjiachen/workspace/short_doc/short_doc_0723_hunhe_v2/混合_v2.0/dataelem_ocr_v2'
    combine_data_to_uie_trainval_txt(src, dst)
