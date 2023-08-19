import shutil
import os
from pathlib import Path
from sklearn.model_selection import train_test_split


def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)


def _train_test_split(dataset_path, oup_path):
    oup_path = Path(oup_path)
    train_path = oup_path / 'train'
    val_path = oup_path / 'val'
    check_folder(train_path)
    check_folder(val_path)

    img_files = list(Path(dataset_path).glob('[!.]*'))

    train_ds, val_ds = train_test_split(
        img_files, train_size=0.9, test_size=0.1, shuffle=True, random_state=144
    )

    for i, train_img in enumerate(train_ds):
        shutil.copy(train_img, train_path)
    print('train num:', i + 1)

    for j, val_img in enumerate(val_ds):
        shutil.copy(val_img, val_path)
    print('val nums:', j + 1)


data_path = '/home/youjiachen/ocr-mrcnn/workspace/datasets/text_det_dataset_v1/Images'
oup_path = '/home/youjiachen/ocr-mrcnn/workspace/datasets/text_det_dataset_v1/'

_train_test_split(data_path, oup_path) 
