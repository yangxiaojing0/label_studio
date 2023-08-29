import json
from pathlib import Path
import shutil

import pandas as pd

def get_all_excel(trainval_folder,oup_excel):
    '''得到所有的预估结果excel'''
    Path(oup_excel).mkdir(parents=True,exist_ok=True)
    excel_list=list(Path(trainval_folder).glob('**/[!.]*.xlsx'))
    for i in excel_list:
        shutil.copy(i,Path(oup_excel)/(i.parent.parent.name+'.xlsx'))
        # if '2023-07-24' in i.name:
        #     shutil.copy(i,Path(oup_excel)/(i.parent.parent.name+'.xlsx'))
            
def get_all_val_res_vis(trainval_folder,save_path):
    val_list=list(Path(trainval_folder).glob('*/vis_prompt_label'))
    for scene_vis_prompt_label in val_list:
        val_res_vis_save_path=Path(save_path)/scene_vis_prompt_label.parent.name
        shutil.copytree(scene_vis_prompt_label,val_res_vis_save_path)
            
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
    '''得到所有场景评估结果表'''
    excels_folder='/home/youjiachen/workspace/yxj/uie_task/0817_23scenes_doc/result/excel'
    scenes_merge_excel='/home/youjiachen/workspace/yxj/uie_task/0817_23scenes_doc/result'
    # get_excel_merge(excels_folder,scenes_merge_excel)
    
    '''得到所有场景评估merge excel'''
    pass

    '''得到可视化结果汇总'''
    trainval_folder='/home/youjiachen/workspace/yxj/uie_task/0817_23scenes_doc/data/0817_23scenes_doc_convert'
    save_path='/home/youjiachen/workspace/yxj/uie_task/0817_23scenes_doc/result/val_res_vis'
    get_all_val_res_vis(trainval_folder,save_path)
    