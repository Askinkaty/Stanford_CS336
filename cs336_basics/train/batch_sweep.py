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
    p = argparse.ArgumentParser(description="Sweep batch sizes and produce learning curves.")
    p.add_argument(
        "--batch-sizes",
        type=int,
        nargs="*",
        default=None,
        help="Explicit batch sizes to test. If omitted, uses powers of 2 from --min-batch to --max-batch.",
    )
    p.add_argument("--min-batch", type=int, default=1, help="Smallest batch size (power-of-2 grid).")
    p.add_argument("--max-batch", type=int, default=512, help="Largest batch size to attempt.")
    p.add_argument("--max-steps", type=int, default=400, help="Training steps per batch size.")
    p.add_argument("--val-interval", type=int, default=100, help="Validate every N steps.")
    p.add_argument("--val-batches", type=int, default=16, help="Validation batches to average.")
    p.add_argument("--val-batch-size", type=int, default=32, help="Batch size used for validation.")
    # LR: either fixed, or linearly scaled from a reference (base_lr, base_batch)
    p.add_argument("--base-lr", type=float, default=3e-3,
                   help="Optimal LR for --base-batch-size. Scaled linearly for other batch sizes.")
    p.add_argument("--base-batch-size", type=int, default=32,
                   help="Reference batch size for --base-lr.")
    p.add_argument("--fixed-lr", type=float, default=None,
                   help="Use this fixed LR for every batch size (disables linear scaling).")
    p.add_argument("--constant-lr", action="store_true",
                   help="Constant LR schedule (no warmup, no cosine decay).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", type=str, default=None, choices=["float32", "float16", "bfloat16"])
    p.add_argument("--out-dir", type=str, default="output/batch_sweep")
    return p.parse_args()


def build_batch_grid(args: argparse.Namespace) -> list[int]:
    if args.batch_sizes:
        return sorted(set(args.batch_sizes))
    sizes = []
    b = args.min_batch
    while b <= args.max_batch:
        sizes.append(b)
        b *= 2
    # ensure typical sizes are included if they fall in range
    for extra in (64, 128):
        if args.min_batch <= extra <= args.max_batch and extra not in sizes:
            sizes.append(extra)
    return sorted(set(sizes))


def lr_for_batch(batch_size: int, args: argparse.Namespace) -> float:
    if args.fixed_lr is not None:
        return args.fixed_lr
    return args.base_lr * batch_size / args.base_batch_size


def quick_validate(trainer: Trainer, val_batch_size: int, num_batches: int) -> float:
    trainer.model.eval()
    device = trainer.cfg.trainer.device
    losses: list[float] = []
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(trainer.valid_dataset.get_iterator(val_batch_size)):
            if idx >= num_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            loss = cross_entropy_loss(trainer.model(inputs), targets).item()
            losses.append(float(loss))
    return float(sum(losses) / len(losses)) if losses else float("inf")


def run_single_batch(
    base_cfg: Config,
    batch_size: int,
    lr: float,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict:
    device = args.device if args.device else base_cfg.trainer.device
    dtype = args.dtype if args.dtype else base_cfg.trainer.dtype

    trainer_cfg = replace(
        base_cfg.trainer,
        output_dir=str(output_dir / f"batch_{batch_size}"),
        max_steps=args.max_steps,
        val_interval=max(1, args.val_interval),
        save_interval=args.max_steps + 1,
        log_interval=args.max_steps + 1,
        seed=args.seed,
        device=device,
        dtype=dtype,
    )
    optim_cfg = replace(base_cfg.optimizer, learning_rate=lr)
    if args.constant_lr:
        optim_cfg = replace(optim_cfg, min_lr=lr, num_warmup_steps=0,
                            cosine_steps=max(1, args.max_steps))
    data_cfg = replace(base_cfg.data, batch_size=batch_size, val_batch_size=args.val_batch_size)
    cfg = replace(base_cfg, trainer=trainer_cfg, optimizer=optim_cfg, data=data_cfg)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        trainer = Trainer(cfg=cfg, wandb=None)
    except torch.cuda.OutOfMemoryError:
        return _oom_result(batch_size, lr)

    initial_val_loss = quick_validate(trainer, args.val_batch_size, args.val_batches)
    best_val_loss = initial_val_loss
    last_val_loss = initial_val_loss
    best_train_loss = float("inf")
    min_train_loss = float("inf")
    first_train_loss = None
    last_train_loss = None

    train_curve: list[dict] = []
    val_curve: list[dict] = [{"step": 0, "val_loss": initial_val_loss}]

    oom = False
    for step in range(args.max_steps):
        trainer.iteration = step
        try:
            inputs, targets = trainer.train_dataset.get_batch(batch_size)
            inputs, targets = inputs.to(device), targets.to(device)
            stats = trainer.train_step(inputs, targets)
        except torch.cuda.OutOfMemoryError:
            oom = True
            break

        train_loss = float(stats["train_loss"])
        train_curve.append({"step": step, "train_loss": train_loss})

        if first_train_loss is None and math.isfinite(train_loss):
            first_train_loss = train_loss
        last_train_loss = train_loss
        best_train_loss = min(best_train_loss, train_loss) if math.isfinite(train_loss) else best_train_loss
        min_train_loss = min(min_train_loss, train_loss) if math.isfinite(train_loss) else min_train_loss

        should_validate = (step + 1) % args.val_interval == 0 or (step + 1) == args.max_steps
        if should_validate:
            try:
                val_loss = quick_validate(trainer, args.val_batch_size, args.val_batches)
            except torch.cuda.OutOfMemoryError:
                oom = True
                break
            val_curve.append({"step": step + 1, "val_loss": val_loss})
            best_val_loss = min(best_val_loss, val_loss)
            last_val_loss = val_loss

    return {
        "batch_size": batch_size,
        "lr": lr,
        "oom": oom,
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


def _oom_result(batch_size: int, lr: float) -> dict:
    return {
        "batch_size": batch_size, "lr": lr, "oom": True,
        "first_train_loss": None, "min_train_loss": None,
        "last_train_loss": None, "best_train_loss": None,
        "initial_val_loss": None, "last_val_loss": None, "best_val_loss": None,
        "train_curve": [], "val_curve": [],
    }


def plot_learning_curves(results: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = [r for r in sorted(results, key=lambda r: r["batch_size"]) if not r["oom"]]
    if not ordered:
        print("No successful runs to plot.")
        return

    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(ordered) - 1, 1)) for i in range(len(ordered))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for r, color in zip(ordered, colors):
        label = f"bs={r['batch_size']} lr={r['lr']:.2e}"

        train_curve = r.get("train_curve", [])
        if train_curve:
            steps = [p["step"] for p in train_curve]
            losses = [p["train_loss"] for p in train_curve]
            axes[0].plot(steps, losses, color=color, linewidth=1.2, label=label)

        val_curve = r.get("val_curve", [])
        if len(val_curve) > 1:
            steps = [p["step"] for p in val_curve]
            losses = [p["val_loss"] for p in val_curve]
            axes[1].plot(steps, losses, color=color, linewidth=1.2, label=label)

    for ax, title in zip(axes, ["Train loss", "Val loss"]):
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Batch size sweep — learning curves", fontsize=12)
    fig.tight_layout()
    path = out_dir / "learning_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {path}")


def write_csv(path: Path, results: list[dict]) -> None:
    fields = [
        "batch_size", "lr", "oom",
        "first_train_loss", "min_train_loss", "last_train_loss", "best_train_loss",
        "initial_val_loss", "last_val_loss", "best_val_loss",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(results, key=lambda r: r["batch_size"]):
            writer.writerow({k: row[k] for k in fields})


def main() -> None:
    args = parse_args()
    batch_sizes = build_batch_grid(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for idx, bs in enumerate(batch_sizes):
        lr = lr_for_batch(bs, args)
        print(f"[{idx + 1}/{len(batch_sizes)}] batch_size={bs} lr={lr:.4g} ...", flush=True)
        result = run_single_batch(default_config, bs, lr, args, out_dir)
        results.append(result)

        status = "OOM" if result["oom"] else f"best_val={result['best_val_loss']:.4f}" if result["best_val_loss"] else "done"
        print(f"  → {status}", flush=True)

        if result["oom"]:
            print(f"  OOM at batch_size={bs}, stopping sweep.", flush=True)
            break

    json_payload = {"args": vars(args), "results": sorted(results, key=lambda r: r["batch_size"])}
    json_path = out_dir / "results.json"
    csv_path = out_dir / "results.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    write_csv(csv_path, results)
    plot_learning_curves(results, out_dir)

    print(f"\nSaved JSON : {json_path}")
    print(f"Saved CSV  : {csv_path}")

    oom_sizes = [r["batch_size"] for r in results if r["oom"]]
    max_ok = max((r["batch_size"] for r in results if not r["oom"]), default=None)
    if oom_sizes:
        print(f"GPU memory limit: OOM at batch_size={oom_sizes[0]}, max working={max_ok}")


if __name__ == "__main__":
    main()