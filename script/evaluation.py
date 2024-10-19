import argparse
import json
import os

from tqdm import tqdm

from safe_sora.metrics import ClipReward, get_psnr, hpsv2_reward


def parse_args():
    parser = argparse.ArgumentParser(description='Traditional evaluation methods for safe-sora')
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--output_path', type=str, default='./outputs/config.json')
    parser.add_argument('--cache_dir', type=str, default='./cache')
    parser.add_argument('--evaluation_mode', type=str, default='psnr', help='psnr, hpsv2 or clip')
    return parser.parse_args()


def load_dataset(dataset_path):
    if not dataset_path.endswith('.json'):
        video_dir = dataset_path
        dataset_path = os.path.join(dataset_path, 'config.json')
    else:
        video_dir = os.path.dirname(dataset_path)
    with open(dataset_path) as f:
        dataset = json.load(f)
    return dataset, video_dir


def main():
    # initialize
    args = parse_args()
    dataset, video_dir = load_dataset(args.dataset_path)
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    # evaluation
    for i in tqdm(range(len(dataset))):
        prompt = dataset[i]['prompt_text']
        video_path0 = os.path.join(video_dir, dataset[i]['video_0']['video_path'])
        video_path1 = os.path.join(video_dir, dataset[i]['video_1']['video_path'])

        if args.evaluation_mode == 'psnr':
            dataset[i]['video_0']['psnr'] = get_psnr(video_path0)
            dataset[i]['video_1']['psnr'] = get_psnr(video_path1)
        elif args.evaluation_mode == 'clip':
            if i == 0:
                reward_model = ClipReward('cuda')
            dataset[i]['video_0']['clip'] = reward_model(prompt, video_path0)
            dataset[i]['video_1']['clip'] = reward_model(prompt, video_path1)
        elif args.evaluation_mode == 'hpsv2':
            dataset[i]['video_0']['hpsv2'] = hpsv2_reward(
                prompt,
                video_path0,
                args.cache_dir,
                sample_rate=0.1,
            )
            dataset[i]['video_1']['hpsv2'] = hpsv2_reward(
                prompt,
                video_path1,
                args.cache_dir,
                sample_rate=0.1,
            )
        else:
            raise ValueError("evaluation_mod should be one of 'psnr', 'clip' or 'hpsv2'")
    # save
    with open(args.output_path, 'w') as f:
        json.dump(dataset, f, indent=4)


if __name__ == '__main__':
    main()
