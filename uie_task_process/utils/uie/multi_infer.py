import json
import subprocess
from pathlib import Path

def infer(
    model_path,
    val_path,
    keys,
):
    subprocess.run(
        [
            'python',
            '/mnt/disk0/youjiachen/PaddleNLP/applications/information_extraction/document/doc_smart_structure/ie_infer.py',
            '--task_path',
            model_path,
            '--val_folder',
            val_path,
            '--keys',
            keys,
        ],
        stdout=subprocess.PIPE,
    )


if __name__ == '__main__':
    '''单场景'''
    # model_path = '/mnt/disk0/youjiachen/workspace/contract/ds_v2.2_混合/dataelem_ocr_v2_negative_ratio_1/checkpoint/model_best'
    # path = '/mnt/disk0/youjiachen/workspace/contract/ds_v2.3_trainval'
    # keys = '项目名称;甲方;乙方;合同生效或失效条款;乙方开户银行;乙方银行账号;合同总价小写;合同总价大写;不含税总价小写;不含税总价大写;签订日期;合同号;税率;合同期限'
    # img_paths = list(Path(path).glob('*/val_images'))
    # for val_path in img_paths:
    #     infer(model_path, val_path, keys)
        
    '''多场景评估, 获得验证集识别结果val_res'''
    model_path='/home/youjiachen/workspace/yxj/uie_task/0817_23scenes_label_match_val/data/0817_23scenes_doc_convert_combine/dataelem_ocr_v2/checkpoint/model_best'
    scence_foler='/home/youjiachen/workspace/yxj/uie_task/0817_23scenes_label_match_val/data/0817_23scenes_doc_convert'
    
    # 获得keys
    scence_list=list(Path(scence_foler).glob('[!.]*'))
    for scence in scence_list:
        print(scence.name+':')
        key_list=[]
        json_path=list(Path(scence).glob('[!.]*_studio.json'))
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
        keys=str(';'.join(keys_list))
        print(keys)
        
        # 进行推理，获得val_res
        img_paths = list(Path(scence).glob('val_images'))
        for val_path in img_paths:
            infer(model_path, val_path, keys)
            # if '道路运输证' in val_path.parent.name:
            #     infer(model_path, val_path, keys)
                



