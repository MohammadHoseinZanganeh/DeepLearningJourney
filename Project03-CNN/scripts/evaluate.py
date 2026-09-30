"""
evaluation module for U-Net semantic segmentation (Exercise 3, Question 1-5).
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from data.data_loader import build_dataloaders, load_config
from models.model import build_model
from utils.metrics import MeanIoU
from utils.losses import build_criterion
from utils.visualization import get_predictions, visualize_predictions


def load_trained_model(
    model_path: Path,
    config: Dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """load a trained model from checkpoint"""
    model = build_model(config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    print(f"  Loaded model from: {model_path}")
    val_miou = checkpoint.get('val_miou', 'unknown')
    if isinstance(val_miou, float):
        print(f"    Best val mIoU: {val_miou:.4f}")
    else:
        print(f"    Best val mIoU: {val_miou}")
    
    return model


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    metric: MeanIoU,
) -> Dict[str, Any]:
    """evaluate model on entire dataset"""
    model.eval()
    metric.reset()
    total_loss = 0.0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            loss = criterion(logits, masks)

            total_loss += loss.item()
            metric.update(logits.argmax(dim=1), masks)

    avg_loss = total_loss / len(loader)
    result = metric.compute()
    
    return {
        "loss": avg_loss,
        "miou": result["miou"],
        "per_class": result["per_class"],
    }


def print_evaluation_results(
    run_name: str,
    result: Dict[str, Any],
    class_names: list,
) -> None:
    """print evaluation results in a nice format"""
    print(f"\n[{run_name}]")
    print(f"  Val Loss: {result['loss']:.4f}")
    print(f"  Val mIoU: {result['miou']:.4f}")
    
    # print per-class IoU for top 3 and bottom 3 classes
    per_class = result['per_class']
    class_ious = list(zip(class_names, per_class))
    class_ious.sort(key=lambda x: x[1], reverse=True)
    
    print("  Per-class IoU (top 3):")
    for name, iou in class_ious[:3]:
        print(f"    {name:15s}: {iou:.4f}")
    
    print("  Per-class IoU (bottom 3):")
    for name, iou in class_ious[-3:]:
        print(f"    {name:15s}: {iou:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate U-Net models")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--model", default=None, help="specific model to evaluate")
    parser.add_argument("--num_vis", type=int, default=5, help="number of samples for visualization")
    args = parser.parse_args()
    
    print("="*60)
    print("  Exercise 3 - Model Evaluation (Question 1-5)")
    print("="*60)
    
    # load config and data
    config = load_config(args.config)
    loaders = build_dataloaders(args.config)
    
    # set device
    if config["training"]["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"  Device: CPU")
    
    class_names = config["classes"]["names"]
    criterion = build_criterion(config, device)
    metric = MeanIoU(num_classes=config["classes"]["num_classes"], ignore_index=9)
    
    # list of all 4 trained models (using results folder)
    configurations = [
        ("bilinear_skipOn", "bilinear", True),
        ("bilinear_skipOff", "bilinear", False),
        ("transposed_skipOn", "transposed", True),
        ("transposed_skipOff", "transposed", False),
    ]
    
    # if specific model requested, only evaluate that one
    if args.model:
        configurations = [(args.model, None, None)]
    
    print("\n" + "="*60)
    print("  EVALUATION RESULTS")
    print("="*60)
    
    results = {}
    
    for run_name, upsample_mode, skip_conn in configurations:
        # update config if needed
        if upsample_mode is not None:
            config["model"]["upsample_mode"] = upsample_mode
            config["model"]["skip_connections"] = skip_conn
        
        # find model path (search in results folder)
        model_path = ROOT / "results" / run_name / "best_model.pth"
        
        if not model_path.exists():
            print(f"\n[{run_name}] Model not found at: {model_path}")
            continue
        
        # load and evaluate
        model = load_trained_model(model_path, config, device)
        result = evaluate_model(model, loaders["val"], criterion, device, metric)
        
        results[run_name] = result
        print_evaluation_results(run_name, result, class_names)
    
    # print summary table
    if results:
        print("\n" + "="*60)
        print("  SUMMARY - mIoU Comparison")
        print("="*60)
        print(f"\n{'Model':<25} {'mIoU':>10}")
        print("-"*37)
        
        best_run = None
        best_miou = -1
        
        for run_name, result in results.items():
            miou = result["miou"]
            print(f"{run_name:<25} {miou:>10.4f}")
            if miou > best_miou:
                best_miou = miou
                best_run = run_name
        
        print("-"*37)
        print(f"\n🏆 Best model: {best_run} (mIoU: {best_miou:.4f})")
    
    # visualize predictions for the best model
    if results and best_run:
        best_model_path = ROOT / "results" / best_run / "best_model.pth"
        
        if best_model_path.exists():
            print(f"\nGenerating visualizations for {best_run}...")
            
            # update config for best model
            for run_name, upsample_mode, skip_conn in configurations:
                if run_name == best_run and upsample_mode is not None:
                    config["model"]["upsample_mode"] = upsample_mode
                    config["model"]["skip_connections"] = skip_conn
                    break
            
            model = load_trained_model(best_model_path, config, device)
            
            images, masks_true, masks_pred = get_predictions(
                model, loaders["val"], device, args.num_vis
            )
            
            vis_path = ROOT / "results" / best_run / "predictions.png"
            visualize_predictions(images, masks_true, masks_pred, class_names, vis_path)
            print(f"  Visualizations saved to: {vis_path}")
    
    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()