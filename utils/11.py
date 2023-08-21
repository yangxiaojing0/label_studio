import shutil
from pathlib import Path

dir = r'F:\desk\08-1标注数据\法院文书-司法公开告知书\图片'
img_list = list(Path(dir).glob('[!.]*'))
for img in img_list:
    save_path = Path(img).parent / str(img.stem)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    shutil.copy(img, save_path)
