import shutil
from pathlib import Path

from tqdm import tqdm

if __name__ == '__main__':
    src = '/mnt/disk0/youjiachen/workspace/ticket-0723'
    dst = '/mnt/disk0/youjiachen/workspace/ticket-0723-excel'
    Path(dst).mkdir(exist_ok=True, parents=True)
    scene_excels = list(Path(src).glob('*/*/*.xlsx'))
    for excel in tqdm(scene_excels):
        scene_name = excel.parents[1].name
        print(scene_name)
        new_name = excel.name.replace('eval_result', scene_name)
        shutil.copy(excel, Path(dst) / new_name)
