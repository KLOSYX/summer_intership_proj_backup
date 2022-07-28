from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import *
from transformers import get_constant_schedule_with_warmup, BertForMaskedLM
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
transformers.logging.set_verbosity_error()

from utils.info_nce_pytorch import InfoNCE


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class PureTextMlmPretrain(pl.LightningModule):
    def __init__(self, 
                 bert_name='hfl/chinese-macbert-base',
                 **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        self.model = BertForMaskedLM.from_pretrained(bert_name, cache_dir='/data/.cache')
        
    def forward(self, tokens):
        outputs = self.model(input_ids=tokens.input_ids,
                             attention_mask=tokens.attention_mask,
                             labels=tokens.labels,
                             return_dict=True)
        return outputs.loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.num_warmup_steps)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
                    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        tokens = batch
        loss = self(tokens)
        self.log_dict({'train_loss': loss})
        return loss
    
    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        tokens = batch
        loss = self(tokens)
        self.log_dict({'val_loss': loss})
              
    @staticmethod
    def add_model_args(parser):
        parser = ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument('--bert_name', type=str, default='hfl/chinese-macbert-base', help='text decoder')
        parser.add_argument('--num_warmup_steps', type=int, default=0, help='number of warmup steps')
        return parser