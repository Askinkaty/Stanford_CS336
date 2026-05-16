"""
Ablation study: SwiGLU vs SiLU feed-forward networks with matched parameter counts.

SwiGLU: 3 matrices (W1, W2, W3) at d_ff=1344  → 3 × 512 × 1344 = 2,064,384 params/layer
SiLU:   2 matrices (W1, W2)    at d_ff=2048  → 2 × 512 × 2048 = 2,097,152 params/layer  (~matched)

Produces two plots (optimal LR and lower LR), each with train/val subplots.
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


OPTIMAL_LR  = 3.51e-4
LOWER_LR    = 7e-5
SILU_D_FF   = 4 * default_config.model.d_model   # 2048, matched to SwiGLU param count


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
    p.add_argument("--out-dir",      type=str,   default="output/ffn_ablation")
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


def run_condition(base_cfg: Config, ffn_type: str, lr: float,
                  args: argparse.Namespace, out_dir: Path) -> dict:
    d_ff = SILU_D_FF if ffn_type == "silu" else base_cfg.model.dim_feedforward
    label = f"{ffn_type}_dff{d_ff}_lr{lr:.2e}"

    device = args.device if args.device else base_cfg.trainer.device
    dtype  = args.dtype  if args.dtype  else base_cfg.trainer.dtype

    data_overrides: dict = dict(batch_size=args.batch_size, val_batch_size=args.batch_size)
    if args.train_path:
        data_overrides["train_path"] = args.train_path
    if args.val_path:
        data_overrides["val_path"] = args.val_path

    cfg = replace(
        base_cfg,
        model=replace(base_cfg.model, ffn_type=ffn_type, dim_feedforward=d_ff),
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
    n_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"  params: {n_params:,}")

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

    return {"label": label, "ffn_type": ffn_type, "d_ff": d_ff, "lr": lr,
            "diverged": diverged, "train_curve": train_curve, "val_curve": val_curve}


def plot_comparison(result_swiglu: dict, result_silu: dict, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13)

    for r, color, ls in [
        (result_swiglu, "#2563EB", "-"),
        (result_silu,   "#16A34A", "--"),
    ]:
        suffix = " (diverged)" if r["diverged"] else ""
        lbl = f"{r['ffn_type'].upper()} d_ff={r['d_ff']}" + suffix

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

    all_results: list[dict] = []

    for lr, lr_label in [(args.optimal_lr, "optimal"), (args.lower_lr, "lower")]:
        results: list[dict] = []
        for ffn_type in ("swiglu", "silu"):
            print(f"\n--- {ffn_type.upper()}, lr={lr:.2e} ---")
            r = run_condition(default_config, ffn_type, lr, args, out_dir)
            results.append(r)
            all_results.append(r)
            status = "DIVERGED" if r["diverged"] else "OK"
            ft = r["train_curve"][-1]["train_loss"] if r["train_curve"] else float("nan")
            fv = r["val_curve"][-1]["val_loss"]     if r["val_curve"]   else float("nan")
            print(f"  {status}  final_train={ft:.4f}  final_val={fv:.4f}")

        plot_comparison(
            results[0], results[1],
            title=f"SwiGLU vs SiLU (param-matched) — LR = {lr:.2e}",
            out_path=out_dir / f"plot_{lr_label}_lr.png",
        )

    with (out_dir / "results.json").open("w") as f:
        json.dump([{k: v for k, v in r.items() if k not in ("train_curve", "val_curve")}
                   for r in all_results], f, indent=2)

    print(f"\nAll results in {out_dir}")


if __name__ == "__main__":
    main()