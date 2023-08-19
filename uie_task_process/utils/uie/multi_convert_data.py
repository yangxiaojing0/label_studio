import subprocess
from pathlib import Path


def convert_data(datadir: Path, save_dir: Path):
    label_train_studio_file = Path(datadir) / 'label_train_studio.json'
    label_val_studio_file = Path(datadir) / 'label_val_studio.json'
    ocr_results = Path(datadir) / 'ocr_results'

    def process_task(
        label_train_studio_file,
        label_val_studio_file,
        ocr_results,
        save_dir,
        negative_ratio='1',
        task_type='ext',
    ):
        subprocess.run(
            [
                'python',
                '/mnt/disk0/youjiachen/PaddleNLP/applications/information_extraction/label_studio_socr_dataset.py',
                '--label_train_studio_file',
                str(label_train_studio_file),
                '--label_val_studio_file',
                str(label_val_studio_file),
                '--ocr_results',
                str(ocr_results),
                '--save_dir',
                str(save_dir),
                '--negative_ratio',
                negative_ratio,
                '--task_type',
                task_type,
            ],
            stdout=subprocess.PIPE,
        )

    process_task(label_train_studio_file, label_val_studio_file, ocr_results, save_dir)


if __name__ == '__main__':
    '''单场景'''
    datadir='/home/youjiachen/workspace/yxj/uie_task/0816_3scenes_task_ocr_convert_combine'
    save_dir = Path(datadir) / 'dataelem_ocr_v2'
    convert_data(datadir, save_dir)
    
    '''多场景'''
    # src = '/home/youjiachen/workspace/yxj/uie_task/0816_3scenes_task_ocr_convert'
    # scene_list = list(Path(src).glob('[!.]*'))
    # for scene in scene_list:
    #     datadir = scene
    #     save_dir = Path(datadir) / 'dataelem_ocr_v2'
    #     convert_data(datadir, save_dir)
