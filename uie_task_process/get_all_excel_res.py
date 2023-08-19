import json
from pathlib import Path
import shutil

import pandas as pd

def get_all_excel(trainval_folder,oup):
    '''得到所有的预估结果excel'''
    trainval_folder='workspace/short_doc/short_doc_0707_trainval'
    oup='/mnt/disk0/youjiachen/workspace/yxj/uie_data_helper/val_res_vis'
    Path(oup).mkdir(parents=True,exist_ok=True)
    excel_list=list(Path(trainval_folder).glob('**/[!.]*.xlsx'))
    for i in excel_list:
        if '2023-07-24' in i.name:
            pass
            # shutil.copy(i,Path(oup)/(i.parent.parent.name+'.xlsx'))
            
def get_all_val_res_vis():
    folder='workspace/short_doc/short_doc_0707_trainval'
    oup='/mnt/disk0/youjiachen/workspace/yxj/uie_data_helper/val_res_vis'
    Path(oup).mkdir(parents=True,exist_ok=True)
    val_list=list(Path(folder).glob('*/val_res'))
    for i in val_list:
        shutil.copytree(i,oup)
            # shutil.copy(i,Path(oup)/(i.parent.parent.name+'.xlsx'))
            
def get_uie_task_res():
    trainval_folder=''
    oup=''
    pass

def get_excel_merge(excels_folder,scenes_merge_excel):
    excel_list=list(Path(excels_folder).glob('[!.]*.xlsx'))
    scenes_data_list=[]
    for excel_file in excel_list:
        scene_data=pd.read_excel(excel_file)
        scene_name=excel_file.stem
        scene_data.insert(0,'scene_name',scene_name)
        scenes_data_list.append(scene_data)
    scenes_data=pd.concat(scenes_data_list)
    scenes_data.to_excel(Path(scenes_merge_excel)/'scenes_merge.xlsx',index=False)


if 0:
    #得到所有key
    json_file='/mnt/disk0/youjiachen/workspace/short_doc/short_doc_5_scene_06_12_trainval/安置房买卖合同/label_train_studio.json'
    with open(json_file,'r',encoding='utf-8') as f:
        data=json.load(f)
    key_=[]
    for i in data:#每个样本文件
        result_list=i['annotations'][0]['result']
        for result in result_list:
            if result['type'] == 'rectanglelabels':
                key_.append(result['value']['rectanglelabels'][0])
        
    # a = jsonpath(data, '$..rectanglelabels')    
    # print(a)

    
if __name__ == '__main__':
    excels_folder='/home/youjiachen/workspace/yxj/uie_task/0817_23scenes_doc/result/excel'
    scenes_merge_excel='/home/youjiachen/workspace/yxj/uie_task/0817_23scenes_doc/result'
    get_excel_merge(excels_folder,scenes_merge_excel)