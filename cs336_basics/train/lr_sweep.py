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
    p.add_argument(
        "--vary-seed-per-lr",
        action="store_true",
        help="Use seed + run_idx for each LR trial (disabled by default for fair LR comparisons).",
    )
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
    device = trainer.cfg.trainer.device

    losses: list[float] = []
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(trainer.valid_dataset.get_iterator(val_batch_size)):
            if idx >= num_batches:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = trainer.model(inputs)
            loss = cross_entropy_loss(outputs, targets).item()
            losses.append(float(loss))

    if not losses:
        return float("inf")
    return float(sum(losses) / len(losses))



def run_single_lr(base_cfg: Config, lr: float, args: argparse.Namespace, run_idx: int, output_dir: Path) -> dict:
    run_seed = args.seed + run_idx if args.vary_seed_per_lr else args.seed
    trainer_cfg = replace(
        base_cfg.trainer,
        output_dir=str(output_dir / f"lr_{lr:.6g}"),
        max_steps=args.max_steps,
        val_interval=max(1, args.val_interval),
        save_interval=args.max_steps + 1,
        log_interval=args.max_steps + 1,
        seed=run_seed,
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

    # --- Track richer diagnostics for "edge of stability" analysis ---
    best_train_loss = float("inf")
    min_train_loss = float("inf")
    first_train_loss = None
    last_train_loss = None

    # IMPORTANT: compute an initial val loss so best_val_loss is never None just because we diverged early
    initial_val_loss = quick_validate(trainer, args.val_batch_size, args.val_batches)
    best_val_loss = initial_val_loss
    last_val_loss = initial_val_loss

    diverged = False
    diverged_step = None
    diverged_reason = ""

    device = cfg.trainer.device
    train_curve: list[dict] = []
    val_curve: list[dict] = [{"step": 0, "val_loss": initial_val_loss}]

    for step in range(args.max_steps):
        trainer.iteration = step

        inputs, targets = trainer.train_dataset.get_batch(cfg.data.batch_size)
        # be explicit about device here too (train_dataset might return CPU tensors)
        inputs = inputs.to(device)
        targets = targets.to(device)

        stats = trainer.train_step(inputs, targets)
        train_loss = float(stats["train_loss"])
        train_curve.append({"step": step, "train_loss": train_loss})

        if first_train_loss is None and math.isfinite(train_loss):
            first_train_loss = train_loss
        last_train_loss = train_loss

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

        # "edge of stability" detector: compare to best seen so far
        if best_train_loss < float("inf") and train_loss > args.explosion_ratio * best_train_loss:
            diverged = True
            diverged_step = step
            diverged_reason = "exploding_relative_to_best"
            break

        best_train_loss = min(best_train_loss, train_loss)
        min_train_loss = min(min_train_loss, train_loss)

        should_validate = (step + 1) % args.val_interval == 0 or (step + 1) == args.max_steps
        if should_validate:
            val_loss = quick_validate(trainer, args.val_batch_size, args.val_batches)
            val_curve.append({"step": step + 1, "val_loss": val_loss})

            if not math.isfinite(val_loss):
                diverged = True
                diverged_step = step
                diverged_reason = "non_finite_val_loss"
                break

            best_val_loss = min(best_val_loss, val_loss)
            last_val_loss = val_loss

    result = {
        "lr": lr,
        "diverged": diverged,
        "diverged_step": diverged_step,
        "diverged_reason": diverged_reason,
        "first_train_loss": first_train_loss,
        "min_train_loss": None if min_train_loss == float("inf") else min_train_loss,
        "last_train_loss": last_train_loss,
        "best_train_loss": None if best_train_loss == float("inf") else best_train_loss,
        "initial_val_loss": None if initial_val_loss == float("inf") else initial_val_loss,
        "last_val_loss": None if last_val_loss == float("inf") else last_val_loss,
        "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
        "train_curve": train_curve,
        "val_curve": val_curve,
    }
    return result


def summarize(results: list[dict]) -> dict:
    ordered = sorted(results, key=lambda r: r["lr"])
    stable = [r for r in ordered if not r["diverged"] and r["best_val_loss"] is not None]
    finite_val = [
        r for r in ordered
        if r["best_val_loss"] is not None and math.isfinite(float(r["best_val_loss"]))
    ]
    diverged = [r for r in ordered if r["diverged"]]

    # Prefer non-diverged runs; if none exist, fall back to any finite-val run.
    if stable:
        best = min(stable, key=lambda r: r["best_val_loss"])
    elif finite_val:
        best = min(finite_val, key=lambda r: r["best_val_loss"])
    else:
        best = None
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


def plot_learning_curves(results: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    ordered = sorted(results, key=lambda r: r["lr"])
    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(ordered) - 1, 1)) for i in range(len(ordered))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for r, color in zip(ordered, colors):
        label = f"lr={r['lr']:.2e}" + (" (div)" if r["diverged"] else "")
        linestyle = "--" if r["diverged"] else "-"

        train_curve = r.get("train_curve", [])
        if train_curve:
            steps = [p["step"] for p in train_curve]
            losses = [p["train_loss"] for p in train_curve]
            axes[0].plot(steps, losses, linestyle=linestyle, color=color, linewidth=1.2, label=label)

        val_curve = r.get("val_curve", [])
        if len(val_curve) > 1:
            steps = [p["step"] for p in val_curve]
            losses = [p["val_loss"] for p in val_curve]
            axes[1].plot(steps, losses, linestyle=linestyle, color=color, linewidth=1.2, label=label)

    for ax, title in zip(axes, ["Train loss", "Val loss"]):
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "learning_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {path}")


def write_csv(path: Path, results: list[dict]) -> None:
    fields = [
        "lr",
        "diverged",
        "diverged_step",
        "diverged_reason",
        "first_train_loss",
        "min_train_loss",
        "last_train_loss",
        "best_train_loss",
        "initial_val_loss",
        "last_val_loss",
        "best_val_loss",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(results, key=lambda r: r["lr"]):
            writer.writerow({k: row[k] for k in fields})


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

    plot_learning_curves(results, out_dir)

    print("\nSummary")
    print(json.dumps(summary, indent=2))
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")


if __name__ == "__main__":
    main()
