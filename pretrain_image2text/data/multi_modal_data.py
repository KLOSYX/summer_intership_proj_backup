from pathlib import Path
from typing import *
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from transformers import AutoTokenizer, AutoFeatureExtractor
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pandas as pd
from PIL import Image

class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class MultiModalDataset(object):
    def __init__(self, file_path, stage='fit'):
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found!")
        data = pd.read_json(file_path, lines=True)
        self.data = data[data.img.apply(len) > 0]
        self.stage = stage
        self.transformer = self.get_simclr_pipeline_transform(224)
        print('total data:', len(self.data))
        
    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply(
                                                  [color_jitter], p=0.8),
                                              transforms.RandomGrayscale(
                                                  p=0.2),
                                              GaussianBlur(
                                                  kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        text = sample["text"]
        img_name = sample["img"]
        img = Image.open(img_name)
        return self.transformer(img), text if self.stage == 'fit' else img
    
    
class Processor(object):
    def __init__(self, img_processor, text_tokenizer, max_length=256) -> None:
        self.img_processor = img_processor
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
    
    def __call__(self, data: list) -> Any:
        imgs, texts = zip(*data)
        pixels = self.img_processor(imgs, return_tensors='pt').pixel_values
        tokens = self.text_tokenizer(list(texts), return_tensors='pt', truncation=True, max_length=self.max_length, padding='max_length')
        return pixels, tokens


class MultiModalData(pl.LightningDataModule):
    def __init__(self, batch_size_per_gpu=4, num_workers=0, visual_processor='openai/clip-vit-base-patch32', text_tokenizer='hfl/chinese-macbert-base', max_length=256, **args) -> None:
        super().__init__()
        self.save_hyperparameters()
        visual_processor = AutoFeatureExtractor.from_pretrained(visual_processor, cache_dir='/data/.cache')
        tokenizer = AutoTokenizer.from_pretrained(text_tokenizer, cache_dir='/data/.cache')
        self.processor = Processor(visual_processor, tokenizer, max_length=max_length)
        
    def setup(self, stage: str = None) -> None:
        if stage is None or stage == 'fit':
            data = MultiModalDataset(self.hparams.train_path, stage=stage)
            data_size = len(data)
            val_size = self.hparams.val_size
            train_size = data_size - val_size
            self.train_dataset, self.valid_dataset = random_split(
                data, [train_size, val_size])
            
        if stage is None or stage == 'test':
            self.test_dataset = MultiModalDataset(self.hparams.test_path, stage=stage)
            
    def train_dataloader(self) -> Any:
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size_per_gpu, num_workers=self.hparams.num_workers, shuffle=True, collate_fn=self.processor)
    
    def val_dataloader(self) -> Any:
        return DataLoader(self.valid_dataset, batch_size=self.hparams.batch_size_per_gpu, num_workers=self.hparams.num_workers, shuffle=False, collate_fn=self.processor)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size_per_gpu, num_workers=self.hparams.num_workers, shuffle=False, collate_fn=self.processor)
    
    @staticmethod
    def add_data_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max_length', type=int, default=256, help='max length of text')
        parser.add_argument('--visual_processor', type=str, default='openai/clip-vit-base-patch32', help='visual processor')
        parser.add_argument('--text_tokenizer', type=str, default='hfl/chinese-macbert-base', help='text tokenizer')
        parser.add_argument('--train_path', type=str,
                            default='/data/clean_raw_text/data/wukong/wukong_zw_with_img_path_clean_valid_img.json', help='train path')
        parser.add_argument('--test_path', type=str,
                    default='/data/clean_raw_text/data/wukong/wukong_zw_with_img_path_clean_valid_img.json', help='test path')
        parser.add_argument('--val_size', type=int, default=6000, help='validation set size')
        return parser
    
    def add_data_info(self):
        cls_token_id = self.processor.text_tokenizer.cls_token_id
        pad_token_id = self.processor.text_tokenizer.pad_token_id
        vocab_size = self.processor.text_tokenizer.vocab_size
        train_data_size = len(self.train_dataset)
        return {'cls_token_id': cls_token_id, 
                'pad_token_id': pad_token_id, 
                'vocab_size': vocab_size, 
                'tokenizer': self.processor.text_tokenizer, 
                'train_data_size': train_data_size,}
    
    
if __name__ == '__main__':
    # debug
    dm = MultiModalData(batch_size_per_gpu=4, num_workers=0, visual_encoder='openai/clip-vit-base-patch32', text_tokenizer='hfl/chinese-macbert-base',
                        train_path='/data/clean_raw_text/data/wukong/wukong_zw_with_img_path_clean_valid_img.json',
                        test_path='/data/clean_raw_text/data/wukong/wukong_zw_with_img_path_clean_valid_img.json',)
    dm.setup('fit')
    dataloader = dm.train_dataloader()
    it = iter(dataloader)
    while True:
        item = next(it)