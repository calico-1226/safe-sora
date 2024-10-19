import json
import os
import random

from gpt_eval import encode_video, gpt_api
from gpt_prompt_new import REGISTRY


def load_data(data_file_path):
    with open(data_file_path, encoding='utf-8') as f:
        return json.load(f)


def process_videos(data, max_frames=16):
    for item in data:
        for video in item['videos']:
            video['base64_frames'] = encode_video(video['video_path'], max_frames)
    return data


def compare_videos(prompts, m, message_process):
    messages = []
    metadata = []  # To store metadata for each comparison
    for prompt in prompts:
        prompt_text = prompt['prompt_text']
        videos = prompt['videos'][:m]  # Select the first `m` videos

        if len(videos) >= 2:
            for i in range(len(videos) - 1):
                for j in range(i + 1, len(videos)):  # Compare each video with the others
                    video_0 = videos[i]
                    video_1 = videos[j]
                    message = message_process(
                        prompt_text,
                        video_0['base64_frames'],
                        video_1['base64_frames'],
                    )
                    messages.append(message)
                    # Store video metadata
                    metadata.append(
                        {
                            'prompt_text': prompt_text,
                            'video_0_id': video_0['video_id'],
                            'video_1_id': video_1['video_id'],
                            'video_0_path': video_0['video_path'],
                            'video_1_path': video_1['video_path'],
                        },
                    )
    return messages, metadata


def main():
    data_file_path = '/home/juntao/Projects/safe-sora/experiments/bias_test_config.json'
    output_dir = '/home/juntao/Projects/safe-sora/experiments/output'
    model_name = 'gpt-4o-2024-05-13'
    task = 'simple_informativeness'
    max_frames = 5
    num_samples = 100
    n = 2  # Num of prompts
    m = 5  # Num of videos per prompts

    # Load and process data
    data = load_data(data_file_path)
    data = process_videos(data, max_frames=max_frames)

    random.seed(42)
    random.shuffle(data)
    selected_prompts = data[3:4]

    # Get task-related functions
    preference_key, message_process, post_process, cache_checker = (
        REGISTRY[task]['preference_key'],
        REGISTRY[task]['message_process'],
        REGISTRY[task]['post_process'],
        REGISTRY[task]['cache_checker'],
    )

    # Generate comparison messages and metadata
    messages, metadata = compare_videos(selected_prompts, m, message_process)

    # Create cache directory (if it doesn't exist)
    cache_dir = os.path.join(output_dir, '.cache')
    os.makedirs(cache_dir, exist_ok=True)

    # Run GPT API to get comparison results
    results = gpt_api(
        model_name=model_name,
        messages=messages,
        num_workers=10,
        post_process=post_process,
        cache_dir=cache_dir,
        cache_checker=cache_checker,
    )

    # Combine results with metadata
    combined_results = []
    for result, meta in zip(results, metadata):
        combined_results.append(
            {
                'result': result,
                'metadata': meta,
                'task_type': task,
            },
        )

    # Save combined results
    with open(os.path.join(output_dir, 'annotation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
