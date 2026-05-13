import argparse
from pathlib import Path

import wandb

from cs336_basics.train.config import Config, default_config
from cs336_basics.train.trainer import Trainer
from cs336_basics.train.logger import logger
from cs336_basics.train.utils import wandb_run_name, dataclass_to_nested_dict, apply_overrides


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-path", type=str, default=None)
    p.add_argument("--validation-path", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--log-interval", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def build_config(args) -> Config:
    overrides = {}
    if args.train_path:
        overrides["data.train_path"] = args.train_path
    if args.validation_path:
        overrides["data.val_path"] = args.validation_path
    if args.device:
        overrides["trainer.device"] = args.device
    if args.dtype:
        overrides["trainer.dtype"] = args.dtype
    if args.batch_size:
        overrides["data.batch_size"] = args.batch_size
    if args.max_steps:
        overrides["trainer.max_steps"] = args.max_steps
    if args.log_interval:
        overrides["trainer.log_interval"] = args.log_interval
    if args.output_dir:
        overrides["trainer.output_dir"] = args.output_dir
    return apply_overrides(default_config, overrides)


def train(cfg: Config):
    run_name: str = wandb_run_name(cfg)[:47]
    logger.info("WandB run name: %s", run_name)
    run = wandb.init(project=cfg.project, name=run_name, config=dataclass_to_nested_dict(cfg))

    trainer = Trainer(cfg, wandb=run)
    trainer.train()
    run.finish()


if __name__ == "__main__":
    args = parse_args()
    cfg = build_config(args)
    train(cfg)