import argparse
import datetime
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.types import Number
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from safe_sora.datasets import PairDataset
from safe_sora.logger import Logger
from safe_sora.models.score_models.modeling_languagebind import LanguageBindForScore
from safe_sora.video_utils import load_video_from_path


class HelpfulnessDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.data = PairDataset.load(
            '/home/juntao/Data/DVG/backup/240p/gpt_preference.json',
            video_dir='/home/juntao/Data/DVG/backup/240p/videos',
        )

        self.input_ids = [
            self.tokenizer(
                x['prompt_text'],
                max_length=77,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )['input_ids'][0]
            for x in tqdm(self.data, desc='Tokenizing', disable=not is_main_process())
        ]
        self.video_set = []
        self.video_path2index = {}
        self.video_0 = [x['video_0']['video_path'] for x in self.data]
        self.video_1 = [x['video_1']['video_path'] for x in self.data]
        self.preference = [x['helpfulness'] for x in self.data]

        for path in tqdm(
            self.video_0 + self.video_1,
            desc='Loading videos',
            disable=not is_main_process(),
        ):
            if path not in self.video_path2index:
                self.video_path2index[path] = len(self.video_set)
                self.video_set.append(load_video_from_path(path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        video = {
            'video_0': self.video_set[self.video_path2index[self.video_0[index]]],
            'video_1': self.video_set[self.video_path2index[self.video_1[index]]],
        }
        preferred = self.preference[index]
        not_preferred = 'video_0' if preferred == 'video_1' else 'video_1'

        return {
            'input_ids': self.input_ids[index],
            'chosen_video': video[preferred],
            'rejected_video': video[not_preferred],
        }


def right_padding(sequences: list[torch.Tensor], padding_value: Number) -> torch.Tensor:
    return pad_sequence(sequences, batch_first=True, padding_value=padding_value)


class PreferenceCollator:

    def __init__(self, tokenizer: PreTrainedTokenizer, device: torch.device):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):

        input_ids = [x['input_ids'] for x in batch]
        attention_mask = [
            input_id.new_ones(input_id.size(), dtype=torch.bool) for input_id in input_ids
        ]
        input_ids = right_padding(input_ids, padding_value=self.tokenizer.pad_token_id)
        attention_mask = right_padding(attention_mask, padding_value=0)

        chosen_video = torch.stack([x['chosen_video'] for x in batch])
        rejected_video = torch.stack([x['rejected_video'] for x in batch])

        return {
            'higher_input_ids': input_ids.to(self.device),
            'higher_attention_mask': attention_mask.to(self.device),
            'higher_videos': chosen_video.to(self.device),
            'lower_input_ids': input_ids.clone().to(self.device),
            'lower_attention_mask': attention_mask.clone().to(self.device),
            'lower_videos': rejected_video.to(self.device),
        }


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def assign_learning_rate(optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    def _warmup_lr(base_lr, warmup_length, step):
        return base_lr * (step + 1) / warmup_length

    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def init_distributed():
    if 'WORLD_SIZE' not in os.environ:
        print('Not in distributed mode.')
        return False

    # Initialize torch.distributed
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        timeout=datetime.timedelta(seconds=180),
    )
    return True


def random_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    args.seed = 100
    args.gradient_checkpointing = True
    args.lr = 5e-4
    args.coef_lr = 1e-3
    args.weight_decay = 0.2
    args.warmup = 200
    args.precision = 'amp_bf16'

    args.epoch = 5
    args.per_device_batch_size = 4

    return args


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def is_main_process():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def barrier():
    if torch.distributed.is_initialized():
        dist.barrier()


def get_all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Perform all-reduce operation on a tensor cross all ranks and return the mean."""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor


def main():

    args = parse_args()

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        raise ValueError('No CUDA device available.')

    init_distributed()
    args.world_size = torch.distributed.get_world_size()
    args.rank = torch.distributed.get_rank()
    args.device = torch.device('cuda', args.rank)
    seed_everything(args.seed * 100 + args.rank)

    barrier()
    print(f'Rank: {args.rank}, World size: {args.world_size}, Device: {args.device}')

    model = LanguageBindForScore(dtype=torch.bfloat16, device=args.device)
    tokenizer = model.tokenizer
    args.image_size = model.languagebind.vision_model.config.image_size

    model.train()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.device],
        find_unused_parameters=True,
    )

    val_model = LanguageBindForScore(dtype=torch.bfloat16, device=args.device)
    val_model.eval()
    # optimizer = model.set_optimizer(args.lr, args.coef_lr, args.weight_decay)
    no_decay = (
        lambda n, p: p.ndim < 2
        or 'bn' in n
        or 'ln' in n
        or 'bias' in n
        or 'logit_scale' in n
        or 'class_embedding' in n
        or 'patch_embedding' in n
    )
    decay = lambda n, p: not no_decay(n, p)

    named_parameters = list(model.named_parameters())
    no_decay_parameters = [p for n, p in named_parameters if no_decay(n, p) and p.requires_grad]
    decay_parameters = [p for n, p in named_parameters if decay(n, p) and p.requires_grad]

    parameter_groups = []
    if len(no_decay_parameters) > 0:
        parameter_groups.append(
            {
                'params': no_decay_parameters,
                'weight_decay': 0.0,
                'lr': args.lr * args.coef_lr,
            },
        )
    if len(decay_parameters) > 0:
        parameter_groups.append(
            {
                'params': decay_parameters,
                'weight_decay': args.weight_decay,
                'lr': args.lr * args.coef_lr,
            },
        )

    optimizer = optim.AdamW(
        parameter_groups,
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-6,
    )

    data = HelpfulnessDataset(tokenizer=tokenizer)
    barrier()

    dataloader = DataLoader(
        data,
        sampler=DistributedSampler(data, shuffle=True),
        collate_fn=PreferenceCollator(tokenizer, args.device),
        batch_size=args.per_device_batch_size,
    )
    total_steps = args.epoch * len(dataloader)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    logger = Logger(
        log_type='wandb',
        log_dir='./outputs/test',
        log_project='test',
        log_run_name='test',
    )

    bar = tqdm(total=total_steps, disable=not is_main_process())
    step = 0
    for epoch in range(args.epoch):
        for batch in dataloader:
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, info = model(**batch)
                logger.print(f'total: {step} - ', info['scores'].clone().detach())
                logger.print(f'high: {step} - ', info['higher_scores'].clone().detach())
                logger.print(f'low: {step} - ', info['lower_scores'].clone().detach())

                with torch.no_grad():
                    _, val_info = val_model(**batch)
                    scores = val_info['scores']
                    logger.print(f'Val: {step} - ', scores)

            loss.backward()
            optimizer.step()
            scheduler(step)

            for key, value in info.items():
                if isinstance(value, torch.Tensor):
                    info[key] = get_all_reduce_mean(value.mean()).item()

            logger.print(f'Step: {step} - ', info)
            logger.log(info, step=step)

            step += 1
            bar.update(1)
    bar.close()
    if is_main_process():
        model.module.save('./outputs/test/')
    logger.close()


if __name__ == '__main__':
    main()
