import shutil
import subprocess
from pathlib import Path


def uie_val(
    model_path,
    test_path,
    output_dir,
    max_seq_len='512',
    per_device_eval_batch_size='16',
    device='gpu:6',
    debug='True',
):
    subprocess.run(
        [
            'python',
            '/mnt/disk0/youjiachen/PaddleNLP_old/applications/information_extraction/document/evaluate.py',
            '--device',
            str(device),
            '--model_path',
            str(model_path),
            '--test_path',
            str(test_path),
            '--output_dir',
            str(output_dir),
            '--label_names',
            'start_positions',
            'end_positions',
            '--max_seq_len',
            max_seq_len,
            '--per_device_eval_batch_size',
            per_device_eval_batch_size,
            '--debug',
            debug,
        ],
        stdout=subprocess.PIPE,
    )

def get_all_excel(trainval_folder,oup_excel):
    '''得到所有的预估结果excel'''
    Path(oup_excel).mkdir(parents=True,exist_ok=True)
    excel_list=list(Path(trainval_folder).glob('**/[!.]*.xlsx'))
    for i in excel_list:
        shutil.copy(i,Path(oup_excel)/(i.parent.parent.name+'.xlsx'))
        # if '2023-07-24' in i.name:
        #     shutil.copy(i,Path(oup_excel)/(i.parent.parent.name+'.xlsx'))

def multi_val(model_path, data_path):
    '''得到分场景的excel,并储存'''
    scene_list = list(Path(data_path).glob('*/dataelem_ocr_v2'))  # **/dev.txt
    for scene in scene_list:
        # if scene.parent.name != '任免公告':
        #     continue
        test_path = Path(scene) / 'dev.txt'
        output_dir = model_path
        uie_val(model_path, test_path, output_dir)
        
if __name__ == '__main__':
    '''获得总/单场景评估excel'''
    scene_path='/home/youjiachen/workspace/yxj/uie_task/0817_23scenes_doc/data/0817_23scenes_doc_convert_combine'
    model_path = Path(scene_path)/'dataelem_ocr_v2/checkpoint/model_best'
    test_path = Path(scene_path)/'dataelem_ocr_v2/dev.txt'
    output_dir = model_path
    # uie_val(model_path, test_path, output_dir)
    
    '''获得每个场景评估excel'''
    model_path = '/home/youjiachen/workspace/yxj/uie_task/0817_23scenes_label_match_val/data/0817_23scenes_doc_convert_combine/dataelem_ocr_v2/checkpoint/model_best'
    data_path = '/home/youjiachen/workspace/yxj/uie_task/0817_23scenes_label_match_val/data/0817_23scenes_doc_convert'
    multi_val(model_path, data_path)
    
    '''摘取所有excel'''
    trainval_folder='/home/youjiachen/workspace/yxj/uie_task/0817_23scenes_label_match_val/data/0817_23scenes_doc_convert'
    oup_excel='/home/youjiachen/workspace/yxj/uie_task/0817_23scenes_label_match_val/result'
    get_all_excel(trainval_folder,oup_excel)
    
    