
    
from collections import defaultdict
import json
import pandas as pd
from pathlib import Path


if __name__ == '__main__':
    scence_foler='/mnt/disk0/youjiachen/workspace/short_doc/short_doc_0707_trainval'
    excel_path='/mnt/disk0/youjiachen/workspace/short_doc/code_for_list_count/uie_doc_excel.xlsx'
    scene_dict=defaultdict(str)
    # 获得keys
    scence_list=list(Path(scence_foler).glob('[!.]*'))
    for scene in scence_list:
        print(scene.name+':')
        key_list=[]
        json_path=list(Path(scene).glob('[!.]*_studio.json'))
        for json_file in json_path:
            with open(json_file,'r',encoding='utf-8') as f1:
                json_data = json.load(f1)
                for i in json_data:#每个样本文件
                    result_list=i['annotations'][0]['result']
                    for result in result_list:
                        if result['type'] == 'rectanglelabels':
                            add_key=result['value']['rectanglelabels'][0]
                            key_list.append(add_key)
        keys_list=list(set(key_list))
        scene_keys=str(';'.join(keys_list))
        print(scene_keys)
        scene_dict[scene.name]=scene_keys
    df=pd.DataFrame.from_dict(scene_dict,orient='index')
    df.to_excel(excel_path)
    
    