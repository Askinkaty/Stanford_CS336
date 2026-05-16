"""
Ablation study: pre-norm vs post-norm transformer.

Pre-norm:  x = x + sublayer(RMSNorm(x))   (+ final RMSNorm before output projection)
Post-norm: x = RMSNorm(x + sublayer(x))   (no extra final norm)

Produces one plot with two subplots (train loss, val loss).
Runs at the optimal LR from the prior sweep (~3.51e-4) and optionally a
second plot at a lower LR to test whether post-norm needs a smaller step size.
"""
import argparse
import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from cs336_basics.train.config import Config, default_config
from cs336_basics.train.loss import cross_entropy_loss
from cs336_basics.train.trainer import Trainer


OPTIMAL_LR = 3.51e-4
LOWER_LR   = 7e-5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--optimal-lr",   type=float, default=OPTIMAL_LR)
    p.add_argument("--lower-lr",     type=float, default=LOWER_LR)
    p.add_argument("--max-steps",    type=int,   default=500)
    p.add_argument("--batch-size",   type=int,   default=8)
    p.add_argument("--val-interval", type=int,   default=50)
    p.add_argument("--val-batches",  type=int,   default=16)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       type=str,   default=None)
    p.add_argument("--dtype",        type=str,   default=None,
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--train-path",   type=str,   default=None)
    p.add_argument("--val-path",     type=str,   default=None)
    p.add_argument("--out-dir",      type=str,   default="output/prenorm_postnorm_ablation")
    return p.parse_args()


def quick_validate(trainer: Trainer, batch_size: int, num_batches: int) -> float:
    trainer.model.eval()
    device = trainer.cfg.trainer.device
    losses: list[float] = []
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(trainer.valid_dataset.get_iterator(batch_size)):
            if idx >= num_batches:
                break
            outputs = trainer.model(inputs.to(device))
            losses.append(float(cross_entropy_loss(outputs, targets.to(device)).item()))
    return float(sum(losses) / len(losses)) if losses else float("inf")


def run_condition(base_cfg: Config, norm_type: str, lr: float,
                  args: argparse.Namespace, out_dir: Path) -> dict:
    label = f"{norm_type}_norm_lr{lr:.2e}"
    device = args.device if args.device else base_cfg.trainer.device
    dtype  = args.dtype  if args.dtype  else base_cfg.trainer.dtype

    data_overrides: dict = dict(batch_size=args.batch_size, val_batch_size=args.batch_size)
    if args.train_path:
        data_overrides["train_path"] = args.train_path
    if args.val_path:
        data_overrides["val_path"] = args.val_path

    cfg = replace(
        base_cfg,
        model=replace(base_cfg.model, norm_type=norm_type),
        optimizer=replace(base_cfg.optimizer, learning_rate=lr, min_lr=0.0,
                          num_warmup_steps=100, cosine_steps=args.max_steps),
        data=replace(base_cfg.data, **data_overrides),
        trainer=replace(base_cfg.trainer,
                        device=device, dtype=dtype,
                        max_steps=args.max_steps,
                        output_dir=str(out_dir / label),
                        save_interval=args.max_steps + 1,
                        log_interval=args.max_steps + 1,
                        seed=args.seed),
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trainer = Trainer(cfg=cfg, wandb=None)

    train_curve: list[dict] = []
    val_curve:   list[dict] = [{"step": 0, "val_loss": quick_validate(trainer, args.batch_size, args.val_batches)}]
    diverged = False

    for step in range(args.max_steps):
        trainer.iteration = step
        inputs, targets = trainer.train_dataset.get_batch(cfg.data.batch_size)
        stats = trainer.train_step(inputs.to(device), targets.to(device))
        tl = float(stats["train_loss"])
        train_curve.append({"step": step, "train_loss": tl})

        if not math.isfinite(tl) or tl > 50.0:
            diverged = True
            break

        if (step + 1) % args.val_interval == 0 or (step + 1) == args.max_steps:
            vl = quick_validate(trainer, args.batch_size, args.val_batches)
            val_curve.append({"step": step + 1, "val_loss": vl})

    return {"label": label, "norm_type": norm_type, "lr": lr,
            "diverged": diverged, "train_curve": train_curve, "val_curve": val_curve}


def plot_comparison(results: list[dict], title: str, out_path: Path) -> None:
    palette = {"pre": "#2563EB", "post": "#16A34A", "none": "#DC2626"}
    linestyle = {"pre": "-", "post": "--", "none": ":"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13)

    for r in results:
        nt = r["norm_type"]
        color = palette.get(nt, "grey")
        ls    = linestyle.get(nt, "-")
        suffix = " (diverged)" if r["diverged"] else ""
        lbl = f"{nt}-norm" + suffix

        tc = r["train_curve"]
        if tc:
            axes[0].plot([p["step"] for p in tc], [p["train_loss"] for p in tc],
                         color=color, ls=ls, lw=1.4, label=lbl)
        vc = r["val_curve"]
        if len(vc) > 1:
            axes[1].plot([p["step"] for p in vc], [p["val_loss"] for p in vc],
                         color=color, ls=ls, lw=1.4, label=lbl)

    for ax, ylabel in zip(axes, ["Train loss", "Val loss"]):
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- optimal LR: pre vs post ---
    results_optimal: list[dict] = []
    for norm_type in ("pre", "post"):
        print(f"\n--- {norm_type}-norm, lr={args.optimal_lr:.2e} ---")
        r = run_condition(default_config, norm_type, args.optimal_lr, args, out_dir)
        results_optimal.append(r)
        status = "DIVERGED" if r["diverged"] else "OK"
        ft = r["train_curve"][-1]["train_loss"] if r["train_curve"] else float("nan")
        fv = r["val_curve"][-1]["val_loss"]     if r["val_curve"]   else float("nan")
        print(f"  {status}  final_train={ft:.4f}  final_val={fv:.4f}")

    plot_comparison(
        results_optimal,
        title=f"Pre-norm vs Post-norm — LR = {args.optimal_lr:.2e}",
        out_path=out_dir / "plot_optimal_lr.png",
    )

    # --- lower LR: pre vs post (tests whether post-norm needs smaller step) ---
    results_lower: list[dict] = []
    for norm_type in ("pre", "post"):
        print(f"\n--- {norm_type}-norm, lr={args.lower_lr:.2e} ---")
        r = run_condition(default_config, norm_type, args.lower_lr, args, out_dir)
        results_lower.append(r)
        status = "DIVERGED" if r["diverged"] else "OK"
        ft = r["train_curve"][-1]["train_loss"] if r["train_curve"] else float("nan")
        fv = r["val_curve"][-1]["val_loss"]     if r["val_curve"]   else float("nan")
        print(f"  {status}  final_train={ft:.4f}  final_val={fv:.4f}")

    plot_comparison(
        results_lower,
        title=f"Pre-norm vs Post-norm — lower LR = {args.lower_lr:.2e}",
        out_path=out_dir / "plot_lower_lr.png",
    )

    # save summary
    all_results = results_optimal + results_lower
    with (out_dir / "results.json").open("w") as f:
        json.dump([{k: v for k, v in r.items() if k not in ("train_curve", "val_curve")}
                   for r in all_results], f, indent=2)

    print(f"\nAll results in {out_dir}")


if __name__ == "__main__":
    main()