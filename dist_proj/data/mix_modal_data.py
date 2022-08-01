from pathlib import Path
from typing import *
from argparse import ArgumentParser
import random

import pytorch_lightning as pl
import torch
from torch import nn
import numpy as np
from transformers import AutoTokenizer, AutoFeatureExtractor, DataCollatorForWholeWordMask
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


class MixModalDataset(object):
    def __init__(self, file_path, stage='fit', image_size=224, multimodal=False, mlm=False, whole_word_mask=False, eda=False, eda_prob=0.5):
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found!")
        self.data = pd.read_json(file_path, lines=True)
        self.stage = stage
        if multimodal:
            self.transformer = self.get_simclr_pipeline_transform(image_size)
        self.multimodal = multimodal
        self.mlm = mlm
        self.whole_word_mask = whole_word_mask
        self.eda = eda
        self.eda_prob = eda_prob
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
        if self.eda and self.stage == 'fit':
            p = random.random()
            if p < self.eda_prob:
                text = random.choice(sample['eda_text'])
        if not self.mlm:
            label = sample["label"]
            if self.multimodal:
                img_name = sample["img"]
                if img_name:
                    img = Image.open(img_name)
                    valid_img = True
                else:
                    # dummy image, will be masked during training
                    img = Image.new('RGB', (224, 224))
                    valid_img = False
                return self.transformer(img), text, valid_img, label
            else:
                return text, label
        else:
            if not self.whole_word_mask:
                return text
            else:
                cn_ref = sample["cn_ref"]
                return text, cn_ref


class Processor(object):
    def __init__(self, img_processor, text_tokenizer, max_length=256, mlm=False, whole_word_mask=False, mlm_probability=0.15) -> None:
        self.img_processor = img_processor
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        self.mlm = mlm
        self.whole_word_mask = whole_word_mask
        self.mlm_probability = mlm_probability

    def __call__(self, data: list) -> Any:
        if not self.mlm:
            if self.img_processor is not None: # multimodal
                imgs, texts, valid_imgs, labels = zip(*data)
                pixels = self.img_processor(imgs, return_tensors='pt')
                tokens = self.text_tokenizer(list(texts), return_tensors='pt', truncation=True,
                                            max_length=self.max_length, padding='max_length')
                valid_imgs = torch.tensor(valid_imgs, dtype=torch.float)
                labels = torch.argmax(torch.tensor(labels, dtype=torch.long), dim=1)
                return pixels, tokens, valid_imgs, labels
            else:
                texts, labels = zip(*data)
                tokens = self.text_tokenizer(list(texts), return_tensors='pt', truncation=True,
                                max_length=self.max_length, padding='max_length')
                labels = torch.argmax(torch.tensor(labels, dtype=torch.long), dim=1)
                return tokens, labels
        else:
            if not self.whole_word_mask:
                tokens = self.text_tokenizer(data, return_tensors='pt', truncation=True,
                                max_length=self.max_length, padding='max_length', return_special_tokens_mask=True)
                tokens["input_ids"], tokens["labels"] = self.torch_mask_tokens(
                    tokens.input_ids, special_tokens_mask=tokens.special_tokens_mask)
                return tokens
            else:
                # whole word mask
                text, cn_refs = zip(*data)
                tokens = self.text_tokenizer(list(text), return_tensors='pt', truncation=True,
                            max_length=self.max_length, padding='max_length')
                mask_labels = [self._whole_word_mask(self._handle_chinese_ref(tokens.tokens(i), cn_refs[i])) for i in range(len(text))]
                mask_labels = torch.tensor(mask_labels, dtype=torch.bool)
                labels = tokens.input_ids.masked_fill(~mask_labels, -100)
                tokens['labels'] = labels
                tokens.input_ids.masked_fill_(mask_labels, self.text_tokenizer.mask_token_id)
                return tokens
                
                
    def _handle_chinese_ref(self, tokens: List[str], cn_ref: List[int]):
        # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
        for i in range(len(tokens)):
            if i in cn_ref:
                tokens[i] = "##" + tokens[i]
        return tokens
                
        
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.text_tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(
                special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.text_tokenizer.convert_tokens_to_ids(
            self.text_tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(
            labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(self.text_tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        cand_indexes = []
        num_valid_token = 0
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]" or token == "[PAD]":
                continue
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
                num_valid_token += 1
            else:
                cand_indexes.append([i])
                num_valid_token += 1

        random.shuffle(cand_indexes)
        # num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        num_to_predict = min(max_predictions, max(1, int(round(num_valid_token * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

class MixModalData(pl.LightningDataModule):
    def __init__(self, tokenizer_name='hfl/chinese-roberta-wwm-ext', 
                 batch_size_per_gpu=4, 
                 num_workers=0, 
                 max_length=256, 
                 multimodal=False, 
                 mlm=False, 
                 whole_word_mask=False, 
                 val_ratio=0.1,
                 eda=False,
                 eda_prob=0.5,
                 **args) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.visual_processor = AutoFeatureExtractor.from_pretrained(
            self.hparams.visual_encoder, cache_dir='/data/.cache') if multimodal else None
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name, cache_dir='/data/.cache')
        self.processor = Processor(
            self.visual_processor, self.tokenizer, max_length=max_length, mlm=mlm, whole_word_mask=whole_word_mask)

    def setup(self, stage: str = None) -> None:
        if stage is None or stage == 'fit':
            data = MixModalDataset(self.hparams.train_path, 
                                   stage=stage,
                                   multimodal=self.visual_processor is not None, 
                                   mlm=self.hparams.mlm, 
                                   whole_word_mask=self.hparams.whole_word_mask, 
                                   eda=self.hparams.eda, 
                                   eda_prob=self.hparams.eda_prob)
            data_size = len(data)
            val_size = int(self.hparams.val_ratio * data_size)
            train_size = data_size - val_size
            print('train_size:', train_size, '\nval_size:', val_size)
            self.train_dataset, self.valid_dataset = random_split(
                data, [train_size, val_size])
            self.valid_dataset.stage = 'val' # change stage to 'val', avoid applying eda on val set

        if stage is None or stage == 'test':
            self.test_dataset = MixModalDataset(
                self.hparams.test_path, 
                stage=stage, 
                multimodal=self.visual_processor is not None,
                mlm=self.hparams.mlm, 
                whole_word_mask=self.hparams.whole_word_mask, 
                eda=self.hparams.eda, 
                eda_prob=self.hparams.eda_prob)

    def train_dataloader(self) -> Any:
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size_per_gpu, num_workers=self.hparams.num_workers, shuffle=True, collate_fn=self.processor, drop_last=True)

    def val_dataloader(self) -> Any:
        return DataLoader(self.valid_dataset, batch_size=self.hparams.batch_size_per_gpu, num_workers=self.hparams.num_workers, shuffle=False, collate_fn=self.processor)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size_per_gpu, num_workers=self.hparams.num_workers, shuffle=False, collate_fn=self.processor)

    @staticmethod
    def add_data_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--tokenizer_name', default='hfl/chinese-roberta-wwm-ext', type=str)
        parser.add_argument('--whole_word_mask', action='store_true', help='Whether to use whole word masking or not.')
        parser.add_argument('--mlm', action='store_true', help='Masked Language Model')
        parser.add_argument('--eda', action='store_true', help='Enable EDA')
        parser.add_argument('--eda_prob', default=0.5, type=float, help='Probability of EDA')
        parser.add_argument('--max_length', type=int,
                            default=256, help='max length of text')
        parser.add_argument('--train_path', type=str,
                            default='/data/clean_raw_text/data/wukong/wukong_zw_with_img_path_clean_valid_img.json', help='train path')
        parser.add_argument('--test_path', type=str,
                            default='/data/clean_raw_text/data/wukong/wukong_zw_with_img_path_clean_valid_img.json', help='test path')
        parser.add_argument('--val_ratio', type=float,
                            default=0.1, help='validation set size')
        parser.add_argument('--multimodal', action='store_true', help='constrative learning')
        return parser


if __name__ == '__main__':
    # debug
    dm = MixModalData(batch_size_per_gpu=4, num_workers=0,
                            train_path='/data/clean_raw_text/all_data_cleaned_with_cn_ref.json',
                            test_path='/data/clean_raw_text/all_data_cleaned_with_cn_ref.json',
                            tokenizer_name='hfl/chinese-roberta-wwm-ext',
                            multimodal=False,
                            mlm=True,
                            whole_word_mask=True,)
    dm.setup('fit')
    dataloader = dm.train_dataloader()
    it = iter(dataloader)
    while True:
        item = next(it)
