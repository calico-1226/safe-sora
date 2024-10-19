import os

import torch
import torch.nn.functional as F

from safe_sora.models.multimodal_encoder.languagebind import (
    LanguageBindVideo,
    LanguageBindVideoTokenizer,
)
from safe_sora.video_utils import get_video_processor


class LanguageBindForScore(torch.nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.pretrained_ckpt = '/home/juntao/Models/LanguageBind/LanguageBind_Video_FT'
        self.languagebind: LanguageBindVideo = LanguageBindVideo.from_pretrained(
            self.pretrained_ckpt,
            cache_dir='./cache_dir',
        )
        self.tokenizer: LanguageBindVideoTokenizer = LanguageBindVideoTokenizer.from_pretrained(
            self.pretrained_ckpt,
            cache_dir='./cache_dir',
        )

        self.dtype = dtype
        self.device = device
        self.regularization = 0.001

        self.languagebind = self.languagebind.to(device, dtype=dtype)
        self.video_process = get_video_processor()

    def train(self):
        self.languagebind.train()

    def eval(self):
        self.languagebind.eval()

    def score(self, videos, prompts):
        text_encoding = self.tokenizer(
            prompts,
            max_length=77,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        # videos: B, C, F, H, W
        video_outputs = self.video_process(videos).to(self.dtype)
        out = self.languagebind(
            input_ids=text_encoding.input_ids.to(self.device),
            pixel_values=video_outputs,
            attention_mask=text_encoding.attention_mask.to(self.device),
        )
        logits = out.text_embeds @ out.image_embeds.T
        scores = torch.diagonal(logits)
        return scores

    def forward(
        self,
        higher_input_ids: torch.LongTensor,
        higher_attention_mask: torch.LongTensor,
        higher_videos: torch.FloatTensor,
        lower_input_ids: torch.LongTensor,
        lower_attention_mask: torch.LongTensor,
        lower_videos: torch.FloatTensor,
    ):  # same as single-frame
        assert higher_input_ids.size(0) == lower_input_ids.size(0), 'Batch size mismatch'
        batch_size = higher_input_ids.size(0)

        higher_pixel_values = self.video_process(higher_videos)
        lower_pixel_values = self.video_process(lower_videos)

        output = self.languagebind(
            input_ids=torch.cat([higher_input_ids, lower_input_ids], dim=0),
            pixel_values=torch.cat([higher_pixel_values, lower_pixel_values], dim=0),
            attention_mask=torch.cat([higher_attention_mask, lower_attention_mask], dim=0),
        )

        logits = output.text_embeds @ output.image_embeds.T
        scores = torch.diagonal(logits)

        higher_scores, lower_scores = scores.chunk(chunks=2, dim=0)

        # higher_output = self.languagebind(
        #     input_ids = higher_input_ids,
        #     pixel_values = higher_pixel_values,
        #     attention_mask = higher_attention_mask,
        # )
        # higher_logits = higher_output.text_embeds @ higher_output.image_embeds.T
        # higher_scores = torch.diagonal(higher_logits)
        # lower_output = self.languagebind(
        #     input_ids = lower_input_ids,
        #     pixel_values = lower_pixel_values,
        #     attention_mask = lower_attention_mask,
        # )
        # lower_logits = lower_output.text_embeds @ lower_output.image_embeds.T
        # lower_scores = torch.diagonal(lower_logits)
        # scores = torch.cat([higher_scores, lower_scores], dim=0)

        loss = -(F.logsigmoid(higher_scores - lower_scores)).mean()

        # pair_scores = torch.stack([torch.stack([higher_scores[i], lower_scores[i]])
        #                            for i in range(0,len(higher_scores))])
        # pair_targets = torch.stack([ torch.tensor([1.0,0.0]).to(pair_scores.device)
        #                             for i in range(0,len(higher_scores))])

        # loss = F.cross_entropy(input=pair_scores, target=pair_targets)

        if self.regularization > 0.0:
            loss = loss + (self.regularization * scores.square().mean())

        # accuracy = (higher_scores > lower_scores).float().mean().detach()
        with torch.no_grad():
            accuracy = (higher_scores > lower_scores).float().mean().detach()
        return loss, {
            'loss': loss,
            'accuracy': accuracy,
            'scores': scores,
            'higher_scores': higher_scores,
            'lower_scores': lower_scores,
        }

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.languagebind.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    def load(self, save_dir: str):
        self.languagebind = LanguageBindVideo.from_pretrained(save_dir)
        self.tokenizer = LanguageBindVideoTokenizer.from_pretrained(save_dir)


if __name__ == '__main__':
    model = LanguageBindForScore(torch.bfloat16, torch.device('cuda:0'))
    print(list(model.named_parameters()))
