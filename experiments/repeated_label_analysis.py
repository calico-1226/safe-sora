import json
from pprint import pprint

import numpy as np
from scipy.special import comb
from tqdm import tqdm


label_path = '/home/juntao/Projects/safe-sora/data/SafeSora-bk/label/label-0604.json'
results = {}

with open(label_path) as f:
    data = json.load(f)

repeat_times2num = {}

for line in data:
    repeated_labels = line['info']['repeated_labels']
    times = str(len(repeated_labels['safe']))
    if times not in repeat_times2num:
        repeat_times2num[times] = 0
    repeat_times2num[times] += 1

pprint(repeat_times2num)

ratio_sum = 0
ratio_count = 0
diff_count = 0

disagree_times = 0
for line in tqdm(data):
    repeated_labels = line['info']['repeated_labels']
    diff_flag = False
    for key, value in repeated_labels.items():
        if key != 'safe':
            continue
        sum_value = np.sum(value)
        num = len(value)
        if num == 1:
            continue
        agreement_ratio = (comb(sum_value, 2) + comb(num - sum_value, 2)) / comb(num, 2)
        ratio_sum += agreement_ratio
        ratio_count += 1
        if sum_value < num:
            diff_flag = True
    if diff_flag:
        diff_count += 1

ratio = ratio_sum / ratio_count
diff_ratio = diff_count / len(data)

print(ratio)
print(diff_ratio)
