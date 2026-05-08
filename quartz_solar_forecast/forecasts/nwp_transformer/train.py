"""Training entry point for nwp_transformer.

CLI usage::

    python -m quartz_solar_forecast.forecasts.nwp_transformer.train \\
        --num-sites 100 --max-epochs 1 --out-dir runs/smoke

Outputs:
    out_dir/config.json: parsed CLI args
    out_dir/train_log.csv: per-epoch train/val loss & MAE
    out_dir/best.pt: best validation-loss checkpoint
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from quartz_solar_forecast.forecasts.nwp_transformer.dataset import (
    QuartzWindowDataset,
    load_site_data,
)
from quartz_solar_forecast.forecasts.nwp_transformer.model import NWPTransformer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train nwp_transformer")
    p.add_argument("--data-dir", type=str, default="data/training")
    p.add_argument(
        "--testset-csv",
        type=str,
        default="quartz_solar_forecast/dataset/testset.csv",
    )
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--num-sites", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--pred-len", type=int, default=96)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def _build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, dict]:
    """Build train/val DataLoaders. Returns (dl_train, dl_val, stats_dict).

    The PV+NWP site arrays are loaded once and shared between the train and
    val Datasets to avoid doubling RAM (which previously triggered OOM kills
    for large ``--num-sites``). Window enumeration is per-split.
    """
    print("[data] loading PV+NWP site arrays…", flush=True)
    t0 = time.time()
    site_data = load_site_data(
        data_dir=args.data_dir,
        testset_csv=args.testset_csv,
        num_sites=args.num_sites,
        seed=args.seed,
        min_valid_steps=args.seq_len + args.pred_len,
    )
    print(
        f"[data] loaded {len(site_data)} sites in {time.time() - t0:.1f}s",
        flush=True,
    )

    common = {
        "site_data": site_data,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "stride": args.stride,
        "seed": args.seed,
    }

    print("[data] enumerating train windows…", flush=True)
    t0 = time.time()
    ds_train = QuartzWindowDataset(split="train", **common)
    print(f"[data] train built in {time.time() - t0:.1f}s — {ds_train.stats}")

    print("[data] enumerating val windows…", flush=True)
    t0 = time.time()
    ds_val = QuartzWindowDataset(split="val", **common)
    print(f"[data] val built in {time.time() - t0:.1f}s — {ds_val.stats}")

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    stats = {"train": ds_train.stats, "val": ds_val.stats}
    return dl_train, dl_val, stats


def _train_one_epoch(
    model: nn.Module,
    dl: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> tuple[float, float]:
    """Train one epoch. Returns (mean_train_loss, lr_at_last_step_used)."""
    model.train()
    n_seen = 0
    loss_sum = 0.0
    last_lr_used = float("nan")
    for x, kwp, y in dl:
        x = x.to(device, non_blocking=True)
        kwp = kwp.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x, kwp)
        loss = criterion(pred, y)

        # Capture the LR that the optimizer is *about to apply* this step.
        # Calling get_last_lr() AFTER scheduler.step() returns the next step's
        # LR, so we record before step.
        last_lr_used = optimizer.param_groups[0]["lr"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        bs = y.size(0)
        loss_sum += loss.item() * bs
        n_seen += bs
    if n_seen == 0:
        return float("nan"), last_lr_used
    return loss_sum / n_seen, last_lr_used


@torch.no_grad()
def _eval_one_epoch(
    model: nn.Module,
    dl: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate one epoch. Returns sample-weighted (mean_loss, mean_abs_err)."""
    model.eval()
    loss_sum = 0.0
    abs_err_sum = 0.0
    n_seen = 0
    for x, kwp, y in dl:
        x = x.to(device, non_blocking=True)
        kwp = kwp.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x, kwp)
        bs = y.size(0)
        loss_sum += criterion(pred, y).item() * bs
        abs_err_sum += (pred - y).abs().mean().item() * bs
        n_seen += bs
    if n_seen == 0:
        return float("nan"), float("nan")
    return loss_sum / n_seen, abs_err_sum / n_seen


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    dl_train, dl_val, ds_stats = _build_loaders(args)
    (out_dir / "data_stats.json").write_text(json.dumps(ds_stats, indent=2))

    if len(dl_train) == 0:
        raise RuntimeError(
            f"Empty train DataLoader (n_windows={len(dl_train.dataset)}, "
            f"batch_size={args.batch_size}, drop_last=True). "
            "Increase --num-sites or decrease --batch-size."
        )

    # Architecture hyperparameters used to instantiate the model. Persisted
    # explicitly so checkpoints stay reconstructable even if model.py
    # defaults shift.
    model_config = {
        "n_nwp_vars": 14,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "d_model": 128,
        "n_heads": 8,
        "e_layers": 3,
        "dropout": 0.2,
        "patch_len": 16,
        "stride": 8,
        "padding": 8,
    }
    device = torch.device(args.device)
    model = NWPTransformer(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params:,} parameters on {device}")

    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    n_steps = max(len(dl_train), 1) * args.max_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=n_steps)
    criterion = nn.MSELoss()

    log_path = out_dir / "train_log.csv"
    log_path.write_text("epoch,train_loss,val_loss,val_mae,lr,duration_s\n")

    best_val = float("inf")
    epochs_since_improve = 0
    for epoch in range(args.max_epochs):
        t_epoch = time.time()
        train_loss, lr = _train_one_epoch(
            model, dl_train, optimizer, scheduler, criterion, device, args.grad_clip,
        )
        val_loss, val_mae = _eval_one_epoch(model, dl_val, criterion, device)
        duration = time.time() - t_epoch

        print(
            f"epoch {epoch + 1}/{args.max_epochs} | "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"val_mae={val_mae:.5f} lr={lr:.2e} ({duration:.1f}s)",
            flush=True,
        )
        with log_path.open("a") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{val_loss:.6f},"
                f"{val_mae:.6f},{lr:.2e},{duration:.1f}\n"
            )

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            epochs_since_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "train_args": vars(args),
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                },
                out_dir / "best.pt",
            )
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(
                    f"[early stop] no improvement for {args.patience} epochs",
                    flush=True,
                )
                break

    print(f"[done] best val_loss={best_val:.5f} -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
