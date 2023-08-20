from collections import defaultdict
from pathlib import Path
from pprint import pprint
import pandas as pd
import yaml


def get_keys_from_yaml(yaml_file_path:str)->str:
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        scene_keys_file = yaml.load(file, Loader=yaml.FullLoader)
    scene_keys=';'.join(scene_keys_file['field_def'])
    return scene_keys
    
    
def get_uie_task_dict_to_excel(scenes_path,excel_path):
    uie_task_dict = defaultdict(str)
    scenes_list=list(Path(scenes_path).glob('[!.]*'))
    for scene in scenes_list:
        yaml_file_path=Path(scene)/'meta.yaml'
        uie_task_dict[str(scene.name).split('-final')[0].split('23-7')[-1]]=get_keys_from_yaml(yaml_file_path)
        pprint(uie_task_dict)
        df=pd.DataFrame.from_dict(uie_task_dict, orient='index')
        df.to_excel(excel_path)
        
if __name__ == '__main__':
    # yaml_file_path='/home/public/ELLM_datasets/卡证表单二期/保险单ps数据-final/meta.yaml'
    # get_keys_from_yaml(yaml_file_path)
    scenes_path='/home/public/ELLM_datasets/smart_structure_idcard_v2.0'
    excel_path='/home/public/ELLM_datasets/code/smart_structure_idcard_v2.0.xlsx'
    get_uie_task_dict_to_excel(scenes_path,excel_path)
    # scene_list=list(Path(scenes_path).glob('[!.]*'))
    # print(len(scene_list))
    
    
    