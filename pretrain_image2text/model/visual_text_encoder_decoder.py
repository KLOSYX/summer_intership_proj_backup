from argparse import ArgumentParser

import pytorch_lightning as pl
import torchmetrics
from transformers import get_constant_schedule_with_warmup
import transformers
import torch
import wandb
transformers.logging.set_verbosity_error()
from einops import rearrange
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from model.layers.multi_modal_model import VisualEncoderLMDecoder

class VisualTextEncoderDecoder(pl.LightningModule):
    def __init__(self, 
                 tokenizer,
                 visual_encoder='google/vit-base-patch16-224-in21k',
                 text_decoder='hfl/chinese-macbert-base',
                 load_mlm_checkpoint=False,
                 **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        # self.tokenizer = self.add_enc_token(tokenizer)
        self.tokenizer = tokenizer
        self.model = VisualEncoderLMDecoder(visual_encoder=visual_encoder, text_decoder=text_decoder, tokenizer=self.tokenizer)
        if load_mlm_checkpoint:
            self.model.load_state_dict(torch.load('/data/proj/pretrain_image2text/pl_log/mixmodal_mlm_pretrain/j1urkq2f/model.ckpt'), strict=False)
        # self.model.text_decoder.resize_token_embeddings(len(self.tokenizer))
        self.bleu = torchmetrics.BLEUScore()
        self.freeze_decoder()
        
    def add_enc_token(self, tokenizer):
        tokenizer.add_special_tokens({'additional_special_tokens': ['<ENC>']})
        tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
        return tokenizer
        
    def forward(self, pixels, tokens):
        loss = self.model(pixels, tokens)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.num_warmup_steps)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
    
    def freeze_decoder(self) -> None:
        for n, p in self.model.named_parameters():
            if n.startswith('visual_encoder.'):
                p.requires_grad = False
                print(f'freeze {n}')
    
    def training_step(self, batch, batch_idx):
        pixels, tokens = batch
        loss = self(pixels, tokens)
        self.log_dict({'train_loss': loss})
        return loss
    
    def validation_step(self, batch, batch_idx):
        pixels, tokens = batch
        loss = self(pixels, tokens)
        generate_ids = self.model.generate(pixels, num_beams=3, max_length=128, min_length=5, no_repeat_ngram_size=2)
        preds = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        targets = self.tokenizer.batch_decode(tokens.input_ids, skip_special_tokens=True)
        targets = [[x] for x in targets]
        if self.hparams.stage == 'fit':
            if not self.hparams.wandb:
                tensorboard = self.logger.experiment
                for i in range(len(preds)):
                    tensorboard.add_text('gen_texts', preds[i], global_step=self.global_step)
                    tensorboard.add_text('ref_texts', targets[i][0], global_step=self.global_step)
            else:
                columns = ['preds', 'image']
                # img_arr : torch.Tensor = rearrange(pixels, 'b c h w -> b h w c')
                data = [[pred, wandb.Image(to_pil_image(pixels[i]), caption=f'{target[0]}')] for i, (pred, target) in enumerate(zip(preds, targets))]
                self.logger.log_text(key=f"samples from epoch {self.current_epoch}", columns=columns, data=data)
        self.log_dict({'val_loss': loss, 'val_bleu': self.bleu(preds, targets)})
        
    @staticmethod
    def add_model_args(parser):
        parser = ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument('--load_mlm_checkpoint', action='store_true', help='load mlm checkpoint or not')
        parser.add_argument('--visual_encoder', type=str, default='google/vit-base-patch16-224-in21k', help='visual encoder')
        parser.add_argument('--text_decoder', type=str, default='hfl/chinese-macbert-base', help='text decoder')
        parser.add_argument('--num_warmup_steps', type=int, default=0, help='number of warmup steps')
        return parser