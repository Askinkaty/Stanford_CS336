import argparse
import csv
import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch

from cs336_basics.train.config import Config, default_config
from cs336_basics.train.loss import cross_entropy_loss
from cs336_basics.train.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep learning rates and measure instability boundary.")
    p.add_argument("--lrs", type=float, nargs="*", default=None, help="Explicit LR values to test.")
    p.add_argument("--min-lr", type=float, default=1e-5, help="Lower bound for log-spaced LR sweep.")
    p.add_argument("--max-lr", type=float, default=3e-2, help="Upper bound for log-spaced LR sweep.")
    p.add_argument("--num-lrs", type=int, default=10, help="Number of log-spaced LR values.")
    p.add_argument("--max-steps", type=int, default=400, help="Train steps per LR.")
    p.add_argument("--batch-size", type=int, default=8, help="Train batch size.")
    p.add_argument("--val-batch-size", type=int, default=8, help="Validation batch size.")
    p.add_argument("--val-interval", type=int, default=100, help="Validate every N steps.")
    p.add_argument("--val-batches", type=int, default=16, help="Validation batches to average.")
    p.add_argument(
        "--constant-lr",
        action="store_true",
        help="Use a constant LR schedule (min_lr=learning_rate, warmup=0).",
    )
    p.add_argument("--seed", type=int, default=42, help="Base random seed.")
    p.add_argument("--device", type=str, default=None, help="Override device.")
    p.add_argument("--dtype", type=str, default=None, choices=["float32", "float16", "bfloat16"], help="Override dtype.")
    p.add_argument(
        "--out-dir",
        type=str,
        default="output/lr_sweep",
        help="Root directory for sweep artifacts.",
    )
    p.add_argument(
        "--train-loss-ceiling",
        type=float,
        default=20.0,
        help="Declare divergence if train loss exceeds this value.",
    )
    p.add_argument(
        "--explosion-ratio",
        type=float,
        default=4.0,
        help="Declare divergence if loss exceeds ratio * best_train_loss_so_far.",
    )
    return p.parse_args()


def build_lr_grid(args: argparse.Namespace) -> list[float]:
    if args.lrs:
        return sorted(set(args.lrs))
    if args.num_lrs < 2:
        return [args.min_lr]
    min_log = math.log10(args.min_lr)
    max_log = math.log10(args.max_lr)
    return [10 ** (min_log + i * (max_log - min_log) / (args.num_lrs - 1)) for i in range(args.num_lrs)]


def quick_validate(trainer: Trainer, val_batch_size: int, num_batches: int) -> float:
    trainer.model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(trainer.valid_dataset.get_iterator(val_batch_size)):
            if idx >= num_batches:
                break
            outputs = trainer.model(inputs)
            loss = cross_entropy_loss(outputs, targets).item()
            losses.append(loss)
    if not losses:
        return float("inf")
    return float(sum(losses) / len(losses))


def run_single_lr(base_cfg: Config, lr: float, args: argparse.Namespace, run_idx: int, output_dir: Path) -> dict:
    trainer_cfg = replace(
        base_cfg.trainer,
        output_dir=str(output_dir / f"lr_{lr:.6g}"),
        max_steps=args.max_steps,
        val_interval=max(1, args.val_interval),
        save_interval=args.max_steps + 1,
        log_interval=args.max_steps + 1,
        seed=args.seed + run_idx,
        device=args.device if args.device else base_cfg.trainer.device,
        dtype=args.dtype if args.dtype else base_cfg.trainer.dtype,
    )
    optim_cfg = replace(
        base_cfg.optimizer,
        learning_rate=lr,
    )
    if args.constant_lr:
        optim_cfg = replace(optim_cfg, min_lr=lr, num_warmup_steps=0, cosine_steps=max(1, args.max_steps))
    data_cfg = replace(base_cfg.data, batch_size=args.batch_size, val_batch_size=args.val_batch_size)
    cfg = replace(base_cfg, trainer=trainer_cfg, optimizer=optim_cfg, data=data_cfg)

    torch.manual_seed(cfg.trainer.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.trainer.seed)

    trainer = Trainer(cfg=cfg, wandb=None)
    best_train_loss = float("inf")
    best_val_loss = float("inf")
    diverged = False
    diverged_step = None
    diverged_reason = ""

    for step in range(args.max_steps):
        trainer.iteration = step
        inputs, targets = trainer.train_dataset.get_batch(cfg.data.batch_size)
        stats = trainer.train_step(inputs, targets)
        train_loss = float(stats["train_loss"])

        if not math.isfinite(train_loss):
            diverged = True
            diverged_step = step
            diverged_reason = "non_finite_train_loss"
            break

        if train_loss > args.train_loss_ceiling:
            diverged = True
            diverged_step = step
            diverged_reason = "train_loss_ceiling"
            break

        if best_train_loss < float("inf") and train_loss > args.explosion_ratio * best_train_loss:
            diverged = True
            diverged_step = step
            diverged_reason = "exploding_relative_to_best"
            break

        best_train_loss = min(best_train_loss, train_loss)

        should_validate = (step + 1) % args.val_interval == 0 or (step + 1) == args.max_steps
        if should_validate:
            val_loss = quick_validate(trainer, args.val_batch_size, args.val_batches)
            best_val_loss = min(best_val_loss, val_loss)
            if not math.isfinite(val_loss):
                diverged = True
                diverged_step = step
                diverged_reason = "non_finite_val_loss"
                break

    result = {
        "lr": lr,
        "diverged": diverged,
        "diverged_step": diverged_step,
        "diverged_reason": diverged_reason,
        "best_train_loss": None if best_train_loss == float("inf") else best_train_loss,
        "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
    }
    return result


def summarize(results: list[dict]) -> dict:
    ordered = sorted(results, key=lambda r: r["lr"])
    stable = [r for r in ordered if not r["diverged"] and r["best_val_loss"] is not None]
    diverged = [r for r in ordered if r["diverged"]]

    best = min(stable, key=lambda r: r["best_val_loss"]) if stable else None
    first_diverged = diverged[0] if diverged else None
    edge_stable = stable[-1] if stable else None

    summary = {
        "num_trials": len(ordered),
        "num_stable": len(stable),
        "num_diverged": len(diverged),
        "best_lr": None if best is None else best["lr"],
        "best_val_loss": None if best is None else best["best_val_loss"],
        "edge_stable_lr": None if edge_stable is None else edge_stable["lr"],
        "first_diverged_lr": None if first_diverged is None else first_diverged["lr"],
        "best_to_edge_ratio": None,
        "best_to_first_diverged_ratio": None,
    }

    if best is not None and edge_stable is not None and edge_stable["lr"] > 0:
        summary["best_to_edge_ratio"] = best["lr"] / edge_stable["lr"]
    if best is not None and first_diverged is not None and first_diverged["lr"] > 0:
        summary["best_to_first_diverged_ratio"] = best["lr"] / first_diverged["lr"]

    return summary


def write_csv(path: Path, results: list[dict]) -> None:
    fields = ["lr", "diverged", "diverged_step", "diverged_reason", "best_train_loss", "best_val_loss"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(results, key=lambda r: r["lr"]):
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    lrs = build_lr_grid(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for idx, lr in enumerate(lrs):
        result = run_single_lr(default_config, lr, args, idx, out_dir)
        results.append(result)
        print(
            f"[{idx + 1}/{len(lrs)}] lr={lr:.6g} "
            f"diverged={result['diverged']} "
            f"best_val_loss={result['best_val_loss']}"
        )

    summary = summarize(results)
    json_payload = {"args": vars(args), "summary": summary, "results": sorted(results, key=lambda r: r["lr"])}

    json_path = out_dir / "results.json"
    csv_path = out_dir / "results.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    write_csv(csv_path, results)

    print("\nSummary")
    print(json.dumps(summary, indent=2))
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")


if __name__ == "__main__":
    main()
