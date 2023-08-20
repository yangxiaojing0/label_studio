from collections import Counter, defaultdict
from pathlib import Path
from pprint import pprint

import pandas as pd


def get_all_format(folder, count=True, son=True):
    """
    寻找文件夹及其各级子文件夹中文件所有后缀名，并统计
    folder：文件夹--》folder = r'F:/desk/sjxs/'（最好一定按照这种格式写）
    count：是否统计每个后缀名出现的次数，并用打印字典
    son:是否包含各级子文件夹，只统计一级子目录--》son=False
    """
    all_format = []

    if son:
        all_files_folders = list(Path(folder).glob('**/[!.]*'))  # 所有：自身文件夹、子文件夹、文件
    else:
        all_files_folders = list(Path(folder).glob('[!.]*'))  # 所有：自身文件夹、子文件夹、文件

    for i in all_files_folders:
        all_format.append(i.suffix)
    # print(all_format)#打印所有文件后缀，不剔除重复项
    if count:
        print(Counter(all_format))
    all_format = set(all_format)  # 剔除重复元素
    all_format = list(all_format)  # 集合形式转为列表形式
    print('后缀名：')
    print(all_format)
    return all_format


# 统计文件数
def files_num(folder_path, excel_path, except_file_format='.json', excel=True):
    """
    计数folder_path的每个一级子文件夹下的所有文件个数，并打印字典
    folder_path:要计数的文件夹--》folder_path = r'F:\desk\sjxs'
    except_file_format：要排除的文件类型，默认为‘json’
    excel：是否保存结果至excel，默认保存
    return：key--》一级子文件夹名称，value--》子文件夹下的各级目录所有文件个数
    """
    print('变量设置：', locals())
    scenes_path = list(Path(folder_path).glob('[!.]*')) 
    stats_dict = defaultdict(int)  # 定义空字典,键是一个整数，值是一个整数。

    for scene in scenes_path:
        files = list(scene.glob('**/[!.]*.*'))
        if len(files) != 0:
            for i in files:
                if i.is_file() and i.suffix != except_file_format:  # 排除文件夹和.json文件
                    # 字典中，key为folder_path中一级子文件夹的名称，value为key文件夹下的文件个数，这个操作使得计数+1
                    stats_dict[scene.name] += 1
        else:
            stats_dict[scene.name] == 0  # 空文件夹
    pprint(
        stats_dict
    )  # pprint的英文全称Data pretty printer,更漂亮地打印,例如：defaultdict(<class 'int'>, {'2file': 1, '4meeting': 4})
    df = pd.DataFrame.from_dict(stats_dict, orient='index')
    if excel:
        df.to_excel(excel_path)
    return stats_dict



if __name__ == '__main__':
    # *****************************************************************************
    # 统计文件数
    files_path = r'/home/public/ELLM_datasets/卡证表单二期'
    excel_path = '/home/public/ELLM_datasets/1num.xlsx'
    # files_num(files_path, except_file_format='.json', excel=True)  # 不统计空文件夹
    files_num(files_path, excel_path, except_file_format='.json', excel=True)
