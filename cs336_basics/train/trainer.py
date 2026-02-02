import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from jaxtyping import Int
from tqdm.auto import tqdm
import logging

from cs336_basics.model.transformer import Transformer
from cs336_basics.train.config import TrainingConfig
from cs336_basics.train.logger import logger
from cs336_basics.train.dataloader import Dataset, load_checkpoint, save_checkpoint
from cs336_basics.model.optimizer import AdamW, get_cosine_lr, gradient_clipping
from cs336_basics.train.loss import cross_entropy_loss
from cs336_basics.train.config import Config, default_config
from cs336_basics.train.utils import load_config, save_config, apply_overrides

torch.set_float32_matmul_precision("high")


class Trainer:
    def __init__(
        self,
        cfg: Config | None = None,
        load_from: str | Path | None = None,
        wandb: Any = None,):
        self.cfg = cfg if cfg is not None else default_config

        self.model = Transformer(
            vocab_size=self.cfg.model.vocab_size,
            context_length=self.cfg.data.context_length,
            d_model=self.cfg.model.d_model,
            num_layers=self.cfg.model.num_layers,
            num_heads=self.cfg.model.n_head,
            d_ff=self.cfg.model.dim_feedforward,
            rope_theta=self.cfg.model.theta,
            device=self.cfg.training.device,
            dtype=getattr(torch, self.cfg.training.dtype),
        )

        self.model.to(self.cfg.training.device)
        logger.info("Model initialized with %d parameters", sum(p.numel() for p in self.model.parameters()))

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.optimizer.learning_rate,
            weight_decay=self.cfg.optimizer.weight_decay,
            betas=self.cfg.optimizer.betas
        )

        self.train_dataset = Dataset(
            path_to_data=self.cfg.data.train_path,
            context_length=self.cfg.data.context_length,
            device=self.cfg.training.device,
        )

        self.valid_dataset = Dataset(
            path_to_data=self.cfg.data.val_path,
            context_length=self.cfg.data.context_length,
            device=self.cfg.training.device,
        )

        self.save_dir = Path(self.cfg.training.output_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.iteration = 0
        self.wandb = wandb

        if load_from is not None:
            self.load_state(load_from)


    def load_state(self, path):
        logger.info("Loading checkpoint from %s", path)
        self.iteration = load_checkpoint(self.model, self.optimizer, path)


    def save_state(self, path):
        logger.info("Saving checkpoint to %s", path)
        save_checkpoint(self.model, self.optimizer, self.iteration, path)


    def get_lr(self, lr, lr_min):
        iter_lr = get_cosine_lr(self.iteration,
                                lr,
                                lr_min,
                                self.cfg.optimizer.num_warmup_steps,
                                self.cfg.optimizer.cosine_steps)
        return iter_lr


    def set_lr(self):
        lr = self.get_lr(self.cfg.optimizer.learning_rate, self.cfg.optimizer.min_lr)
        for p in self.optimizer.param_groups:
            p["lr"] = lr
        return lr

    def log(self, **data):
        for k, v in data.items():
            if "log" not in k:
                logger.info(f"{k}: {v}")
        if self.wandb is not None:
            self.wandb.log(data)

    def train_step(self, inputs, targets):
        self.model.train()
        iter_lr = self.set_lr()

        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = cross_entropy_loss(outputs, targets)

        if self.cfg.trainer.gradient_accumulation_steps > 1:
            loss /= self.cfg.trainer.gradient_accumulation_steps

        loss.backward()
        gradient_clipping(self.model.parameters(), self.cfg.training.max_grad_norm)
        self.optimizer.step()

        return {"loss": loss.item(), "lr": iter_lr}


    def train(self):
        logger.info("Starting training for %d epochs", self.cfg.training.num_epochs)
        while self.iteration < self.cfg.training.max_steps:
            if self.iteration % self.cfg.training.save_interval == 0:
                checkpoint_path = self.save_dir / f"checkpoint_iter_{self.iteration}.pt"
                self.save_state(checkpoint_path)

            epoch_start_time = time.time()
            inputs, targets = self.train_dataset.get_batch(self.cfg.data.batch_size)
            stats = self.train_step(inputs, targets)
            self.iteration += 1
            self.save_state(checkpoint_path)
            epoch_end_time = time.time()


    def validate(self):
        self.model.eval()
        val_iters = 0
        val_loss_epoch = torch.zeros((), device=self.model.embeddings.weight.device, dtype=torch.float32)
        for inputs, targets in tqdm(self.valid_dataset.get_iterator(self.cfg.data.val_batch_size),
            total=len(self.valid_dataset) // self.cfg.data.val_batch_size):
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = cross_entropy_loss(outputs, targets)
                val_loss_epoch += loss.to(torch.float32)
                val_iters += 1
        val_loss_epoch = (val_loss_epoch / val_iters).item()
        return {"val_loss": val_loss_epoch, "val_ppl": math.exp(val_loss_epoch)}


    def generate(self, input_ids: Int[Tensor, "seq"],
                 eos_token_id: int,
                 max_new_tokens: int = 20,
                 temperature: float = 1.0, top_p: float = 1.0) -> Int[Tensor, "seq+new_seq"]:
        self.model.eval()
        return self.model.generate(input_ids, max_new_tokens, eos_token_id, temperature, top_p)
