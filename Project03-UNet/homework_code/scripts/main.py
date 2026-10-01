"""
main script to train U-Net for Exercise 3

usage:
    python scripts/main.py --upsample bilinear --skip true
    python scripts/main.py --upsample transposed --skip false --bn true
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# add project root to path
ROOT: Path = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from data.data_loader import build_dataloaders, load_config
from scripts.train import train_one_config
from utils.visualization import plot_training_history


def parse_args() -> argparse.Namespace:
    """
    parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train U-Net for football segmentation")
    parser.add_argument(
        "--upsample",
        choices=["bilinear", "transposed"],
        required=True,
        help="upsample mode (bilinear or transposed)"
    )
    parser.add_argument(
        "--skip",
        choices=["true", "false"],
        required=True,
        help="skip connections (true or false)"
    )
    parser.add_argument(
        "--bn",
        choices=["true", "false"],
        default="false",
        help="use batch normalization (true or false)"
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "config.yaml"),
        help="path to config file"
    )
    return parser.parse_args()


def save_config_info(run_dir: Path, upsample_mode: str, skip_bool: bool, use_bn: bool, best_miou: float, epochs: int) -> None:
    """
    save configuration info to a text file.
    """
    # make human readable description
    if upsample_mode == "bilinear":
        upsample_text = "Bilinear Upsample + Conv (3x3)"
    else:
        upsample_text = "ConvTranspose2d (stride=2)"
    
    skip_text = "with Skip Connections" if skip_bool else "without Skip Connections"
    bn_text = "with BatchNorm" if use_bn else "without BatchNorm"
    run_name = f"{upsample_mode}_skip{'On' if skip_bool else 'Off'}_bn{'On' if use_bn else 'Off'}"
    
    info_path = run_dir / "config_info.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("MODEL CONFIGURATION\n")
        f.write("="*60 + "\n")
        f.write(f"Run name: {run_name}\n")
        f.write(f"Upsample mode: {upsample_mode}\n")
        f.write(f"Skip connections: {skip_bool}\n")
        f.write(f"Batch Normalization: {use_bn}\n")
        f.write("-"*60 + "\n")
        f.write(f"Description: {upsample_text} - {skip_text} - {bn_text}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Best validation mIoU: {best_miou:.4f}\n")
        f.write(f"Total epochs trained: {epochs}\n")
        f.write("="*60 + "\n")
    
    print(f"  Config info saved to: {info_path}")


def main() -> None:
    """
    main function to train one configuration.
    """
    # parse arguments
    args = parse_args()
    
    skip_bool = args.skip == "true"
    use_bn = args.bn == "true"
    run_name = f"{args.upsample}_skip{'On' if skip_bool else 'Off'}_bn{'On' if use_bn else 'Off'}"
    
    # readable header
    if args.upsample == "bilinear":
        upsample_text = "Bilinear Upsample + Conv (3x3)"
    else:
        upsample_text = "ConvTranspose2d (stride=2)"
    
    skip_text = "with Skip Connections" if skip_bool else "without Skip Connections"
    bn_text = "with BatchNorm" if use_bn else "without BatchNorm"
    
    print("="*60)
    print(f"  Exercise 3 - Football Semantic Segmentation")
    print("="*60)
    print(f"\n  Training: {upsample_text}")
    print(f"  {skip_text}")
    print(f"  {bn_text}")
    print(f"  Run name: {run_name}")
    print("="*60)

    # load config
    print("\n[1] Loading configuration...")
    config: Dict[str, Any] = load_config(args.config)
    
    # add use_bn to config for model building
    config["model"]["use_batch_norm"] = use_bn

    # build dataloaders
    print("[2] Building dataloaders...")
    loaders: Dict[str, Any] = build_dataloaders(args.config)

    # set paths - using results folder
    results_dir: Path = ROOT / "results" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # train
    print(f"\n[3] Training {run_name}...")
    result: Dict[str, Any] = train_one_config(
        base_config=config,
        upsample_mode=args.upsample,
        skip_connections=skip_bool,
        loaders=loaders,
        run_dir=results_dir,
    )

    # save config info
    epochs = config["training"]["epochs"]
    save_config_info(results_dir, args.upsample, skip_bool, use_bn, result["best_miou"], epochs)

    # plot training curves
    history_path = results_dir / "history.json"
    if history_path.exists():
        print("\n[4] Plotting training curves...")
        with open(history_path, "r") as f:
            history = json.load(f)
        
        plot_training_history(
            history=history,
            save_path=results_dir / "training_curves.png"
        )
        print(f"  Curves saved to: {results_dir / 'training_curves.png'}")
    else:
        print("\n[4] Warning: history.json not found")

    # print final result
    print("\n" + "="*60)
    print(f"  TRAINING COMPLETED")
    print("="*60)
    print(f"  Configuration: {run_name}")
    print(f"  Best val mIoU: {result['best_miou']:.4f}")
    print(f"  Results saved to: {results_dir}")
    print("="*60)
    print("\nDone!")


if __name__ == "__main__":
    main()