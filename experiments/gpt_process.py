import json
import os

from gpt_prompt import REGISTRY


def extract_numbers_from_file(file_path):
    # 创建一个空字典用于存储提取的数字
    result_dict = {}

    # 打开文件并逐行读取内容
    with open(file_path) as file:
        for line in file:
            # 使用split方法将每行分为key和value部分
            parts = line.split(':')
            if len(parts) == 2:
                key = parts[0].strip()  # 去除key的前后空白
                value = float(parts[1].strip())  # 将value转换为浮点数并去除前后空白
                result_dict[key] = value

    return result_dict


scr_dir = 'experiments/output/gpt_annotation/'
tasks = REGISTRY.keys()
results = {}
task_in_dir = os.listdir(scr_dir)
model_list = ['gpt-4o-mini', 'gpt-4o-2024-05-13', 'gpt-4o-2024-08-06']
max_frames_list = [1, 2, 4, 8]

for task in tasks:
    if task not in task_in_dir:
        continue
    results[task] = {}
    for model_name in model_list:
        results[task][model_name] = {}
        for max_frames in max_frames_list:
            output_dir = os.path.join(scr_dir, task, model_name, str(max_frames))
            path = os.path.join(output_dir, 'agreement.txt')
            result = extract_numbers_from_file(path)
            result['Agreement ratio'] = (
                result['Agreement count'] + (100 - result['All count']) / 2
            ) / 100
            results[task][model_name][max_frames] = result


with open('experiments/output/gpt_annotation/agreement.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
