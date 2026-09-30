"""
training module for U-Net semantic segmentation (Exercise 3, Question 1-4)

contains all the training functions.
"""

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data.data_loader import load_config
from models.model import build_model
from utils.metrics import MeanIoU
from utils.losses import build_criterion


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    metric: MeanIoU,
) -> Dict[str, float]:
    """run one training epoch on all training data"""
    # set model to training mode (enables dropout, batch norm, etc)
    model.train()
    # reset the metric before starting
    metric.reset()
    total_loss = 0.0

    # loop through all batches in the training loader
    for imgs, masks in loader:
        # move data to the device (GPU or CPU)
        imgs = imgs.to(device)
        masks = masks.to(device)

        # zero out the gradients from previous batch
        optimizer.zero_grad()
        # forward pass: get predictions from model
        logits = model(imgs)
        # calculate loss between predictions and ground truth
        loss = criterion(logits, masks)
        # backward pass: compute gradients
        loss.backward()
        # update model weights
        optimizer.step()

        # add current batch loss to total
        total_loss += loss.item()
        # update metric with predictions
        metric.update(logits.argmax(dim=1), masks)

    # calculate average loss across all batches
    avg_loss = total_loss / len(loader)
    # get the metric result (mIoU)
    result = metric.compute()
    return {"loss": avg_loss, "miou": result["miou"]}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    metric: MeanIoU,
) -> Dict[str, Any]:
    """run validation on the validation set (no gradient computation)"""
    # set model to evaluation mode (no dropout, no batch norm updates)
    model.eval()
    # reset the metric
    metric.reset()
    total_loss = 0.0

    # loop through all batches in validation loader
    for imgs, masks in loader:
        # move data to device
        imgs = imgs.to(device)
        masks = masks.to(device)

        # forward pass (no gradients needed due to decorator)
        logits = model(imgs)
        # calculate loss
        loss = criterion(logits, masks)

        # accumulate loss
        total_loss += loss.item()
        # update metric with predictions
        metric.update(logits.argmax(dim=1), masks)

    # calculate average loss and metrics
    avg_loss = total_loss / len(loader)
    result = metric.compute()
    return {"loss": avg_loss, "miou": result["miou"], "per_class": result["per_class"]}


def train_one_config(
    base_config: Dict[str, Any],
    upsample_mode: str,
    skip_connections: bool,
    loaders: Dict[str, DataLoader],
    run_dir: Path,
) -> Dict[str, Any]:
    """train one U-Net configuration with given upsample mode and skip connection setting"""
    # create directory for saving results (if not exists)
    run_dir.mkdir(parents=True, exist_ok=True)

    # make a deep copy of the config to avoid modifying the original
    cfg = deepcopy(base_config)
    # update config with current settings
    cfg["model"]["upsample_mode"] = upsample_mode
    cfg["model"]["skip_connections"] = skip_connections

    # print header for this training run
    print(f"\n{'='*60}")
    print(f"  Training: upsample={upsample_mode}  skip={skip_connections}")
    print(f"{'='*60}")

    # set device (GPU if available, otherwise CPU)
    if cfg["training"]["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"  Device: CPU (GPU not available)")

    # build the model from config
    model = build_model(cfg).to(device)
    # build the loss function from config
    criterion = build_criterion(cfg, device)
    # create Adam optimizer with learning rate from config
    optimizer = Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    
    # learning rate scheduler: reduces learning rate when mIoU stops improving
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    
    # create metric for mIoU calculation
    metric = MeanIoU(num_classes=cfg["classes"]["num_classes"], ignore_index=9)

    # get training parameters from config
    epochs = cfg["training"]["epochs"]
    best_miou = 0.0
    history: List[Dict[str, Any]] = []

    # main training loop over epochs
    for epoch in range(1, epochs + 1):
        # record start time for this epoch
        t0 = time.time()

        # train for one epoch
        train_m = train_one_epoch(model, loaders["train"], criterion, optimizer, device, metric)
        # validate after training
        val_m = validate(model, loaders["val"], criterion, device, metric)

        # update learning rate based on validation mIoU
        scheduler.step(val_m["miou"])
        # calculate elapsed time for this epoch
        elapsed = time.time() - t0

        # print progress for current epoch
        print(
            f"  Epoch {epoch:>3}/{epochs} | "
            f"train loss={train_m['loss']:.4f}  miou={train_m['miou']:.4f} | "
            f"val loss={val_m['loss']:.4f}  miou={val_m['miou']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # if this is the best model so far, save it
        if val_m["miou"] > best_miou:
            best_miou = val_m["miou"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": deepcopy(model.state_dict()),
                    "val_miou": best_miou,
                    "config": cfg,
                },
                run_dir / "best_model.pth",
            )
            print(f"  -> New best val mIoU: {best_miou:.4f}  (saved)")

        # store history for this epoch
        history.append({
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_miou": train_m["miou"],
            "val_loss": val_m["loss"],
            "val_miou": val_m["miou"],
            "per_class": val_m["per_class"],
        })

    # save training history to json file
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # print final result
    print(f"\n  Finished. Best val mIoU: {best_miou:.4f}")
    return {"best_miou": best_miou, "history": history}