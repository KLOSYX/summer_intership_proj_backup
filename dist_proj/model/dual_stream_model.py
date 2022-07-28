from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import *
from transformers import get_constant_schedule_with_warmup
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
transformers.logging.set_verbosity_error()

from model.layers.multi_modal_model import VisualEncoderMLMEncoder
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

class DualStreamModel(pl.LightningModule):
    def __init__(self, 
                 visual_encoder='google/vit-base-patch16-224-in21k',
                 text_decoder='hfl/chinese-macbert-base',
                 queue_size=57600,
                 momentum=0.995,
                 **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        self.model = VisualEncoderMLMEncoder(visual_encoder=visual_encoder, text_decoder=text_decoder)
        self.model_m = VisualEncoderMLMEncoder(visual_encoder=visual_encoder, text_decoder=text_decoder)
        self.projector = nn.Linear(self.model.text_decoder.config.hidden_size, 256)
        self.projector_m = nn.Linear(self.model_m.text_decoder.config.hidden_size, 256)
        self.model_pairs = [
            [self.model, self.model_m],
            [self.projector, self.projector_m],
        ]
        self.momentum = momentum
        self.copy_params()
        self.register_buffer("queue", torch.randn(queue_size, 256))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size  
        self.criterion = InfoNCE(negative_mode='unpaired', temperature=0.07)
        if self.hparams.freeze_encoder:
            self.freeze_decoder()
        
    def forward(self, img, text, valid_img):
        loss_mlm, last_hidden_state = self.model(img.pixel_values, text, valid_img)
        return loss_mlm, last_hidden_state
    
    def forward_m(self, img, text, valid_img, is_train=True):
        # momentum constractive training
        view1, view2 = torch.chunk(img.pixel_values, 2, dim=0)
        loss_mlm, last_hidden_state = self.model(view1, text, valid_img)
        global_repr = F.normalize(self.projector(last_hidden_state[:, 0, :]), dim=1) # [N, 256]
        with torch.no_grad():
            if is_train:
                self._momentum_update()
            _, last_hidden_state_m = self.model_m(view2, text, valid_img)
            pos_global_repr_m = F.normalize(self.projector_m(last_hidden_state_m[:, 0, :]), dim=1) # [N, 256]
            neg_global_repr_m = self.queue.clone().detach() # [queue_size, 256]
        loss_contrastive = self.criterion(global_repr, pos_global_repr_m, neg_global_repr_m)
        if is_train:
            self._dequeue_and_enqueue(pos_global_repr_m)
        loss = loss_mlm + loss_contrastive
        return loss, (loss_mlm, loss_contrastive)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.num_warmup_steps)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
    
    def freeze_decoder(self):
        for n, p in self.model.named_parameters():
            if n.startswith('visual_encoder.'):
                p.requires_grad = False
                print(f'freeze {n}')
                    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        img, text, valid_img = batch
        loss, (loss_mlm, loss_contrastive) = self.forward_m(img, text, valid_img)
        self.log_dict({'train_loss': loss, 'train_loss_mlm': loss_mlm, 'train_loss_contrastive': loss_contrastive}, batch_size=self.hparams.batch_size_per_gpu)
        return loss
    
    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        img, text, valid_img = batch
        loss, (loss_mlm, loss_contrastive) = self.forward_m(img, text, valid_img, is_train=False)
        self.log_dict({'val_loss': loss, 'val_loss_mlm': loss_mlm, 'val_loss_contrastive': loss_contrastive}, batch_size=self.hparams.batch_size_per_gpu)
        
        
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    
            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, feat):
        # gather keys before updating queue
        feats = concat_all_gather(feat)
        bs = feats.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % bs == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + bs, :] = feats
        ptr = (ptr + bs) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr 
            
    @staticmethod
    def add_model_args(parser):
        parser = ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument('--freeze_encoder', action='store_true', help='freeze encoder')
        parser.add_argument('--visual_encoder', type=str, default='google/vit-base-patch16-224-in21k', help='visual encoder')
        parser.add_argument('--text_decoder', type=str, default='hfl/chinese-macbert-base', help='text decoder')
        parser.add_argument('--num_warmup_steps', type=int, default=0, help='number of warmup steps')
        parser.add_argument('--queue_size', type=int, default=57600, help='queue size')
        return parser