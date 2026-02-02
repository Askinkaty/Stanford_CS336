import argparse
import json
from pathlib import Path

import wandb
from typing import Any
from datetime import datetime
from dataclasses import asdict, replace

from cs336_basics.train.config import Config
from cs336_basics.train.trainer import Trainer
from cs336_basics.train.logger import logger
from cs336_basics.train.utils import wandb_run_name, dataclass_to_nested_dict

# from gpt_small import cfg
from cs336_basics.train.config import default_config as cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train-path",
        type=str,
        default=str(Path(__file__).parent.parent / "data_tokenized/TinyStoriesV2-GPT4-train.npy"),
    )
    p.add_argument(
        "--validation-path",
        type=str,
        default=str(Path(__file__).parent.parent / "data_tokenized/TinyStoriesV2-GPT4-valid.npy"),
    )
    return p.parse_args()


def train(cfg: Config, args):
    run_name: str = wandb_run_name(cfg)[:47]
    logger.info("WandB run name: %s", run_name)
    run = wandb.init(project=cfg.project, name=run_name, config=dataclass_to_nested_dict(cfg))

    trainer = Trainer(
        cfg,
        wandb=run
    )
    trainer.train()
    run.finish()


# def test(cfg: Config):
#     import torch
#
#     trainer = Trainer(cfg)
#     print(trainer.generate(torch.tensor([[0, 1, 2]]), 5, top_p=0.8, temperature=0.1))
#
# train(cfg, parse_args())
# test(cfg)

if __name__ == "__main__":
    train(cfg, parse_args())