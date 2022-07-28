from pathlib import Path
from typing import *
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer, ViTFeatureExtractor
from torch.utils.data import DataLoader, random_split
import pandas as pd
from PIL import Image


class MultiModalDataset(object):
    def __init__(self, file_path):
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found!")
        data = pd.read_json(file_path, lines=True)
        self.data = data[data.img.apply(len) > 0]
        print('total data:', len(self.data))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        text = sample["text"]
        img_name = sample["refine_img"]
        img = Image.open(img_name)
        return img, text
    
    
class Processor(object):
    def __init__(self, img_processor) -> None:
        self.img_processor = img_processor    
        
    def __call__(self, data: list) -> Any:
        imgs, texts = zip(*data)
        pixels = self.img_processor(imgs, return_tensors='pt')
        return pixels, texts


class BlipData(pl.LightningDataModule):
    def __init__(self, batch_size_per_gpu=4, num_workers=0, val_size=6000, visual_processor='google/vit-base-patch16-224-in21k',  **args) -> None:
        super().__init__()
        self.save_hyperparameters()
        visual_processor = ViTFeatureExtractor.from_pretrained(visual_processor, cache_dir='/data/.cache')
        self.processor = Processor(visual_processor)
        
    def setup(self, stage: str = None) -> None:
        if stage is None or stage == 'fit':
            data = MultiModalDataset(self.hparams.train_path)
            data_size = len(data)
            val_size = self.hparams.val_size
            train_size = data_size - val_size
            self.train_dataset, self.valid_dataset = random_split(
                data, [train_size, val_size])
            
        if stage is None or stage == 'test':
            self.test_dataset = MultiModalDataset(self.hparams.test_path)
            
    def train_dataloader(self) -> Any:
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size_per_gpu, num_workers=self.hparams.num_workers, shuffle=True, collate_fn=self.processor)
    
    def val_dataloader(self) -> Any:
        return DataLoader(self.valid_dataset, batch_size=self.hparams.batch_size_per_gpu, num_workers=self.hparams.num_workers, shuffle=False, collate_fn=self.processor)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size_per_gpu, num_workers=self.hparams.num_workers, shuffle=False, collate_fn=self.processor)
    
    @staticmethod
    def add_data_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--visual_processor', type=str, default='google/vit-base-patch16-224-in21k', help='visual processor')
        parser.add_argument('--train_path', type=str,
                            default='/data/clean_raw_text/data/wukong/wukong_zw_with_img_path_clean_valid_img_refine.json', help='train path')
        parser.add_argument('--test_path', type=str,
                    default='/data/clean_raw_text/data/wukong/wukong_zw_with_img_path_clean_valid_img_refine.json', help='test path')
        parser.add_argument('--val_size', type=int, default=6000, help='validation set size')
        return parser
    
    # def add_data_info(self):
    #     cls_token_id = self.processor.text_tokenizer.cls_token_id
    #     pad_token_id = self.processor.text_tokenizer.pad_token_id
    #     vocab_size = self.processor.text_tokenizer.vocab_size
    #     train_data_size = len(self.train_dataset)
    #     return {'cls_token_id': cls_token_id, 
    #             'pad_token_id': pad_token_id, 
    #             'vocab_size': vocab_size, 
    #             'tokenizer': self.processor.text_tokenizer, 
    #             'train_data_size': train_data_size,}
    
    
if __name__ == '__main__':
    # debug
    dm = BlipData(batch_size_per_gpu=4, num_workers=0, visual_encoder='google/vit-base-patch16-224-in21k', 
                        train_path='/data/clean_raw_text/data/wukong/wukong_zw_with_img_path_clean_valid_img_refine.json',
                        test_path='/data/clean_raw_text/data/wukong/wukong_zw_with_img_path_clean_valid_img_refine.json',)
    dm.setup('fit')
    dataloader = dm.train_dataloader()
    it = iter(dataloader)
    while True:
        item = next(it)