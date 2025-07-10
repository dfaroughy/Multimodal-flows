import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from scipy.special import gammaln, gamma, factorial

from tensorclass import TensorMultiModal
from datamodules.datasets import DataCoupling
from transformers import GPT2Config, GPT2LMHeadModel

class JetFlavorSeqGPT(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.max_seq_length = config.max_seq_length    # real tokens per jet
        self.lr = config.lr
        self.lr_final = config.lr_final
        self.temperature = config.temperature
        self.top_k = config.top_k

        # special tokens:
        self.start_token = config.vocab_size + 1
        self.end_token = config.vocab_size + 2
        self.pad_token = config.vocab_size + 3   

        config_gpt = GPT2Config(
            vocab_size=self.pad_token + 1,          # token vocab + BOS + EOS + pads
            n_positions=config.max_seq_length + 2,  # seq with BOS and EOS enpoints
            n_ctx=config.max_seq_length + 2,        # seq with BOS and EOS enpoints
            n_embd=config.n_embd,
            n_inner=config.n_inner if config.n_inner is not None else 4 * config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            activation_function=config.activation,
            attn_pdrop=config.dropout_att,
            embd_pdrop=config.dropout_emb,
            resid_pdrop=config.dropout_res,
            bos_token_id=self.start_token,
            eos_token_id=self.end_token,
            pad_token_id=self.pad_token,
        )

        self.model = GPT2LMHeadModel(config_gpt)
        self.config = config
        self.save_hyperparameters()

    #...train/inference methods

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, 
                          attention_mask=attention_mask
                          ).logits

    def training_step(self, batch: DataCoupling, batch_idx):
        loss = self.model(input_ids=batch.target.discrete,
                          attention_mask=batch.target.mask,
                          labels=self._mask_pads(batch.target.discrete),
                          ).loss

        self.log("train_loss",
                 loss,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 batch_size=len(batch)
                 )

        return {"loss": loss}

    def validation_step(self, batch: DataCoupling, batch_idx):
        loss = self.model(input_ids=batch.target.discrete,
                          attention_mask=batch.target.mask,
                          labels=self._mask_pads(batch.target.discrete),
                          ).loss

        self.log("val_loss",
                 loss,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True,
                 sync_dist=True,
                 batch_size=len(batch)
                 )

        return {"val_loss": loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        sample = self.model.generate(input_ids=batch.source.discrete,
                                    max_new_tokens=self.max_seq_length + 2, 
                                    do_sample=True,
                                    temperature=self.temperature,
                                    top_k = self.top_k,
                                    bos_token_id=self.start_token,
                                    eos_token_id=self.end_token,
                                    pad_token_id=self.pad_token,
                                    )

        sample = F.pad(sample, (0, self.max_seq_length + 2 - sample.shape[1]), value=self.pad_token)
        sample = torch.where(sample >= self.start_token, 0, sample)[:, 1:-1]  # remove start/end tokens

        return sample.detach().cpu()
 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,    # full cycle length
            eta_min=self.lr_final             # final LR
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",   
                "frequency": 1,
                "strict": True,
            },
        }


    def _mask_pads(self, labels):
        """ Mask out the padding tokens in the labels.
        """
        labels = labels.clone()
        labels[labels == self.pad_token] = -100  # CE ignores
        return labels
