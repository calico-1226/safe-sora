import base64
import json
import logging
import os
import random
import time
from typing import Any, Callable, Optional

import cv2
import ray
import urllib3
from gpt_prompt import REGISTRY
from tqdm import tqdm
from urllib3.util.retry import Retry

from safe_sora.datasets import PairDataset
from safe_sora.utils import generate_hash_uid


API_KEY = 'sk-Zbf9diCiUl2kIhPyF71fA5Ab1c6847Fe9f67018c6cAb9cDb'
API_URL = 'https://api.61798.cn/v1'


def bean_gpt_api(
    model_name: str,
    message: dict,
    post_process: Callable = lambda x: x,
) -> Any:
    """Baichuan GPT API"""

    # openai_api = 'https://apejhvxcd.cloud.sealos.io'
    openai_api = 'https://api.61798.cn'

    params_gpt = {
        'model': model_name,
        'messages': message,
        'temperature': 1.0,
    }
    url = openai_api + '/v1/chat/completions'

    headers = {
        'Content-Type': 'application/json',
        'Authorization': API_KEY,
        'Connection': 'close',
    }

    retry_strategy = Retry(
        total=5,  # Maximum retry count
        backoff_factor=0.1,  # Wait factor between retries
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to force a retry on
        allowed_methods=['POST'],  # Retry only for POST request
        raise_on_redirect=False,  # Don't raise exception
        raise_on_status=False,  # Don't raise exception
    )
    timeout = urllib3.Timeout(total=50.0)
    http = urllib3.PoolManager(
        retries=retry_strategy,
        timeout=timeout,
    )
    encoded_data = json.dumps(params_gpt).encode('utf-8')
    max_try = 10
    while max_try > 0:
        try:
            response = http.request('POST', url, body=encoded_data, headers=headers)
            if response.status == 200:
                response = json.loads(response.data.decode('utf-8'))['choices'][0]['message'][
                    'content'
                ]
                logging.info(response)
                break
            else:
                err_msg = f'Access openai error, status code: {response.status} response: {response.data.decode("utf-8")}'
                logging.error(err_msg)
                time.sleep(1)
                max_try -= 1
                continue
        except Exception as e:
            err_msg = f'Access openai error: {e}'
            logging.error(err_msg)
            time.sleep(1)
            max_try -= 1
            continue
    else:
        print('Bean Proxy API Failed...')
        # print('Using OpenAI API...')
        # response = ray.get(gpt_api.remote(system_content, user_content))
        response = 'Bean Proxy API Failed...'

    return post_process(response)


@ray.remote(num_cpus=1)
def _bean_gpt_api(
    model_name: str,
    message: dict,
    post_process: Callable = lambda x: x,
) -> Any:
    return bean_gpt_api(model_name, message, post_process)


def gpt_api(
    model_name: str,
    messages: list[dict],
    num_workers: int = 10,
    post_process: Callable = lambda x: x,
    cache_dir: Optional[str] = None,
    cache_checker: Callable = lambda _: True,
    **kwargs: Any,
):
    """API"""
    api_interaction_count = 0
    ray.init()

    messages = list(enumerate(messages))
    bar = tqdm(total=len(messages))
    results = [None] * len(messages)
    uids = [generate_hash_uid(message) for message in messages]
    not_finished = []

    while True:
        if len(not_finished) == 0 and len(messages) == 0:
            break

        while len(not_finished) < num_workers and len(messages) > 0:
            index, message = messages.pop()
            uid = uids[index]

            if cache_dir is not None:
                cache_path = os.path.join(cache_dir, f'{uid}.json')
                if os.path.exists(cache_path):
                    with open(cache_path, encoding='utf-8') as f:
                        try:
                            result = json.load(f)
                        except json.decoder.JSONDecodeError:
                            print(f'JSONDecodeError: {cache_path}')
                            exit()
                    if cache_checker(result):
                        results[index] = result
                        bar.update(1)
                        continue

            future = _bean_gpt_api.remote(model_name, message, post_process)
            not_finished.append([index, future])
            api_interaction_count += 1

        if len(not_finished) == 0:
            continue

        # Break into a list of indices and a list of futures
        indices, futures = zip(*not_finished)

        finished, not_finished_futures = ray.wait(list(futures), timeout=1.0)

        # Find the index of completed tasks
        finished_indices = [indices[futures.index(task)] for task in finished]

        for i, task in enumerate(finished):
            results[finished_indices[i]] = ray.get(task)
            uid = uids[finished_indices[i]]
            cache_path = os.path.join(cache_dir, f'{uid}.json')
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(results[finished_indices[i]], f, ensure_ascii=False, indent=4)

        # Update the not_finished list to remove completed tasks
        not_finished = [(index, future) for index, future in not_finished if future not in finished]

        bar.update(len(finished))
    bar.close()

    # It is very important to ensure that all results have been collected
    assert all(result is not None for result in results)

    ray.shutdown()
    print(f'API interaction count: {api_interaction_count}')

    return results


def encode_video(video_path, max_frames=16):
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        base64Frames.append(base64.b64encode(buffer).decode('utf-8'))

    video.release()
    interval = max(1, len(base64Frames) // max_frames)
    base64Frames = base64Frames[::interval][:max_frames]

    return base64Frames


def annotate_preference(
    model_name='gpt-4o-2024-05-13',
    task='simple_instruction_following',
    num_samples=100,
    max_frames=4,
    output_dir='/home/juntao/Projects/safe-sora/experiments/output',
):
    data = PairDataset.load(
        '/home/juntao/Data/DVG/240p/config/pairs/pair_config1.json',
        video_dir='/home/juntao/Data/DVG/240p/videos',
    )
    random.seed(42)
    data.shuffle()
    data = data[:num_samples]
    for line in tqdm(data, desc='Encoding videos'):
        line['video_0']['base64frames'] = encode_video(line['video_0']['video_path'], max_frames)
        line['video_1']['base64frames'] = encode_video(line['video_1']['video_path'], max_frames)

    preference_key, message_process, post_process, cache_checker = (
        REGISTRY[task]['preference_key'],
        REGISTRY[task]['message_process'],
        REGISTRY[task]['post_process'],
        REGISTRY[task]['cache_checker'],
    )

    messages = [
        message_process(
            line['prompt_text'],
            line['video_0']['base64frames'],
            line['video_1']['base64frames'],
        )
        for line in data
    ]
    results = gpt_api(
        model_name,
        messages,
        num_workers=10,
        post_process=post_process,
        cache_dir=os.path.join(output_dir, '.cache'),
        cache_checker=cache_checker,
    )
    assert len(messages) == len(results)

    for line, result in zip(data, results):
        # if result[preference_key] is not None:
        #     num_all += 1
        #     if preference_key in ['helpfulness', 'harmlessness']:
        #         if line[preference_key] == result[preference_key]:
        #             num_agreement += 1
        #     elif line['sub_preferences'][preference_key] == result[preference_key]:
        #         num_agreement += 1

        if 'gpt_preference' not in line:
            line['gpt_preference'] = {}
        line['gpt_preference'][preference_key] = result[preference_key]
        line[preference_key] = result[preference_key]

        del line['video_0']['base64frames']
        del line['video_1']['base64frames']

    with open(os.path.join(output_dir, 'annotation.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main() -> None:
    for model_name in ['gpt-4o-2024-05-13']:
        for max_frames in [4]:
            for task in ['helpfulness']:
                NUM_SAMPLES = -1
                MAX_FRAMES = max_frames
                MODEL_NAME = model_name
                TASK = task
                OUTPUT_DIR = (
                    f'/home/juntao/Data/DVG/240p/config/pairs/{TASK}_{MODEL_NAME}_{MAX_FRAMES}_1019'
                )
                os.makedirs(OUTPUT_DIR, exist_ok=True)

                print('=' * 80)
                print(
                    f'Annotating {MODEL_NAME} with {NUM_SAMPLES} samples and {MAX_FRAMES} frames...',
                )
                print('=' * 80)

                for _ in range(5):
                    annotate_preference(
                        model_name=MODEL_NAME,
                        task=TASK,
                        num_samples=NUM_SAMPLES,
                        max_frames=MAX_FRAMES,
                        output_dir=OUTPUT_DIR,
                    )


# Useful for debug
if __name__ == '__main__':
    main()
