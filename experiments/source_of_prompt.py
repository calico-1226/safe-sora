import json
import random
from pprint import pprint

from tqdm import tqdm

from safe_sora.datasets import VideoDataset
from safe_sora.utils import generate_hash_uid


label_data_0 = VideoDataset.load(
    '/home/juntao/Projects/safe-sora/data/SafeSora-Label/config-train.json.gz',
)
label_data_1 = VideoDataset.load(
    '/home/juntao/Projects/safe-sora/data/SafeSora-Label/config-test.json.gz',
)
label_data = VideoDataset(label_data_0.configs + label_data_1.configs)


print(label_data.num_prompts)
print(len(label_data.configs))


data_point = {
    'prompt_id': None,
    'prompt_text': None,
    'prompt_type': None,
    'video_labels': [],
}

data = []
prompt_id2idx = {}

for video in tqdm(label_data, desc='Processing label data'):
    prompt_id = video['prompt_id']
    if prompt_id not in prompt_id2idx:
        prompt_id2idx[prompt_id] = len(data)
        data.append(
            {
                'prompt_id': prompt_id,
                'prompt_text': video['prompt_text'],
                'prompt_type': video['prompt_type'],
                'video_labels': [],
            },
        )
    idx = prompt_id2idx[prompt_id]
    assert data[idx]['prompt_id'] == video['prompt_id']
    assert data[idx]['prompt_text'] == video['prompt_text']
    assert data[idx]['prompt_type'] == video['prompt_type']
    data[idx]['video_labels'].append(video)

# print(len(data))

for prompt in tqdm(data, desc='Collecting refined text'):
    prompt['refined_text'] = set()
    for video_label in prompt['video_labels']:
        if video_label['video_text'] != prompt['prompt_text']:
            prompt['refined_text'].add(video_label['video_text'])
    del prompt['video_labels']

# pprint(data[0])


def get_flatten_data(data):
    flatten_data = []
    hash_set = set()
    for prompt in tqdm(data, desc='Flattening data'):
        # print(prompt['prompt_id'])
        # print(generate_hash_uid(prompt['prompt_text']))
        # assert generate_hash_uid(prompt['prompt_text']) == prompt['prompt_id']
        hash_set.add(prompt['prompt_id'])
        flatten_data.append(
            {
                'prompt_id': prompt['prompt_id'],
                'prompt_text': prompt['prompt_text'],
                'prompt_type': prompt['prompt_type'],
                'source': None,
                'refined_from': None,
            },
        )
        for refined_text in prompt['refined_text']:
            prompt_id = generate_hash_uid(refined_text)
            if prompt_id not in hash_set:
                hash_set.add(prompt_id)
                flatten_data.append(
                    {
                        'prompt_id': prompt_id,
                        'prompt_text': refined_text,
                        'prompt_type': prompt['prompt_type'],
                        'source': 'AI-refined',
                        'refined_from': prompt['prompt_id'],
                    },
                )
    return flatten_data


data = get_flatten_data(data)

day2label = {
    '0508': 'real-user-prompt (VidProM)',
    '0509': 'real-user-prompt (VidProM)',
    '0510': 'researcher-constructed-prompt (midjourney-detailed-prompts)',
    '0512': 'researcher-constructed-prompt (midjourney-detailed-prompts)',
    '0514': 'laion',
    '0515': 'laion',
    '0516': 'real-user-prompt (VidProM)',
    '0517-0': 'laion',
    '0517-1': 'real-user-prompt (Discord)',
    '0521': 'researcher-constructed-prompt (midjourney-detailed-prompts)',
    '0522-0': 'researcher-constructed-prompt (HarmCategory)',
}


def get_day2prompt_id():

    day2path = {
        day: f'/home/juntao/Projects/safe-sora/data/SafeSora-bk/to-annotation-json/to-annotation-{day}.json'
        for day in day2label
    }
    day2prompt_id = {day: set() for day in day2label}
    for day, path in day2path.items():
        with open(path) as f:
            json_data = json.load(f)
        for prompt in json_data:
            day2prompt_id[day].add(prompt['prompt_id'])
    return day2prompt_id


day2prompt_id = get_day2prompt_id()

all = 0
laion_idx = []

for idx, prompt in tqdm(enumerate(data), total=len(data), desc='Checking prompt_id'):

    for day, prompt_id_set in day2prompt_id.items():
        if prompt['prompt_id'] in prompt_id_set:
            assert prompt['source'] is None or prompt['source'] == day2label[day]
            prompt['source'] = day2label[day]
    assert prompt['source'] is not None

    if prompt['source'] == 'laion':
        laion_idx.append(idx)

    all += 1

choice_idx = random.sample(laion_idx, 2203)

for idx in laion_idx:
    if idx not in choice_idx:
        data[idx]['source'] = 'researcher-constructed-prompt (LAION-400M)'
    else:
        data[idx]['source'] = 'real-user-prompt (Discord)'

count = {}

for prompt in data:
    if prompt['source'] not in count:
        count[prompt['source']] = 0
    count[prompt['source']] += 1

print(count)

from datasets import Dataset


hf_data = Dataset.from_list(data)
for line in hf_data:
    if line['refined_from'] is not None:
        pprint(line)
        break

hf_data.to_json('./prompt.jsonl.gz', orient='records', lines=True, compression='gzip')
