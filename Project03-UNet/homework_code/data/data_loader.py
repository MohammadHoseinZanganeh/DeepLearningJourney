"""
data loader module for exercise 3 - football semantic segmentation

this module reads two independent datasets with different structures,
unifies the labels, and prepares them for training.
"""

import os
import re
import numpy as np
import yaml
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# helper functions for reading config

def load_config(config_path: str) -> dict:
    """reads the yaml config file and returns it."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_color_map(color_to_unified: dict) -> dict:
    """
    converts color-to-class mapping from yaml format to usable dictionary.

    input:  {"[237, 34, 236]": 0, ...}
    output: {(237, 34, 236): 0, ...}
    """
    parsed = {}
    for key, val in color_to_unified.items():
        # convert string "[r, g, b]" to tuple (r, g, b)
        nums = re.findall(r"\d+", key)
        rgb = tuple(int(n) for n in nums)
        parsed[rgb] = int(val)
    return parsed


# convert rgb mask to class mask

def build_color_lookup(color_map: dict) -> np.ndarray:
    """
    build a flat lookup table mapping packed RGB int -> class id.

    each RGB triplet (r, g, b) is packed into a single int32:
        key = r * 65536 + g * 256 + b   (max value: 255*65536+255*256+255 = 16,777,215)

    this allows O(1) vectorized lookup over all pixels at once,
    instead of looping over each color in the map per call.

    args:
        color_map: dict {(r, g, b): class_id}

    returns:
        numpy array of shape (16,777,216,) — index is packed rgb key, value is class id
    """
    lut = np.zeros(16_777_216, dtype=np.int64)  # default class = 0 (background)
    for (r, g, b), class_id in color_map.items():
        key = int(r) * 65536 + int(g) * 256 + int(b)
        lut[key] = class_id
    return lut


def mask_rgb_to_class(mask_rgb: np.ndarray, color_map: dict, num_classes: int,
                      lut: np.ndarray = None) -> np.ndarray:
    """
    converts rgb mask (H, W, 3) to class mask (H, W).

    uses a pre-built lookup table for fully vectorized lookup —
    no Python loop over colors, one numpy index operation over all pixels.

    pixels with colors not in the map get class 0 (background).

    args:
        mask_rgb:    numpy array with shape (H, W, 3)
        color_map:   dictionary {(r,g,b): class_id}
        num_classes: total number of classes
        lut:         pre-built lookup table from build_color_lookup().
                     if None, one is built on the fly (slower).

    returns:
        numpy array with shape (H, W) with integer class ids
    """
    if lut is None:
        lut = build_color_lookup(color_map)

    # pack each pixel's (r, g, b) into a single integer key
    keys = (mask_rgb[:, :, 0].astype(np.int32) * 65536 +
            mask_rgb[:, :, 1].astype(np.int32) * 256   +
            mask_rgb[:, :, 2].astype(np.int32))

    # single vectorized lookup over entire mask — no Python loop
    return lut[keys]


# collect image-mask pairs from each dataset

def collect_dataset1_pairs(images_dir: str, masks_dir: str) -> list:
    """find image-mask pairs in dataset1. both have same filename in different folders"""
    pairs = []
    images_path = Path(images_dir)
    masks_path = Path(masks_dir)

    for img_file in images_path.iterdir():
        # only accept image files
        if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        mask_file = masks_path / img_file.name
        if mask_file.exists():
            pairs.append((str(img_file), str(mask_file), 1))

    return pairs


def collect_dataset2_pairs(images_dir: str) -> list:
    """
    collects image and corresponding fuse mask paths from second dataset.

    dataset2 structure:
        images/ -> Frame 1 (1).jpg  and  Frame 1 (1).jpg___fuse.png
        (image and mask are in the same folder)

    mask files are identified by "___fuse.png" suffix.

    output:
        list of tuples (image_path, mask_path, dataset_number)
    """
    pairs = []
    dir_path = Path(images_dir)

    # find all fuse masks
    fuse_files = {f for f in dir_path.iterdir() if f.name.endswith("___fuse.png")}

    for fuse_file in sorted(fuse_files):
        # image name = mask name without "___fuse.png"
        img_name = fuse_file.name.replace("___fuse.png", "")
        img_file = dir_path / img_name

        if img_file.exists():
            pairs.append((str(img_file), str(fuse_file), 2))

    return pairs


# main dataset class

class FootballSegDataset(Dataset):
    """
    football semantic segmentation dataset.

    handles both datasets together and converts labels
    to unified class space.

    arguments:
        pairs:        list of tuples (image_path, mask_path, dataset_number)
        color_map1:   color to class mapping for dataset1
        color_map2:   color to class mapping for dataset2
        num_classes:  number of unified classes
        image_size:   output image size (square)
        augment:      apply data augmentation (only for training)
    """

    def __init__(
        self,
        pairs: list,
        color_map1: dict,
        color_map2: dict,
        num_classes: int,
        image_size: int = 256,
        augment: bool = False,
    ):
        self.pairs = pairs
        self.color_map1 = color_map1
        self.color_map2 = color_map2
        self.num_classes = num_classes
        self.image_size = image_size
        self.augment = augment

        # build lookup tables once at dataset creation — reused in every __getitem__
        self.lut1 = build_color_lookup(color_map1)
        self.lut2 = build_color_lookup(color_map2)

        # convert image to normalized tensor
        self.img_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path, dataset_id = self.pairs[idx]

        # read image
        image = Image.open(img_path).convert("RGB")

        # read mask
        mask_img = Image.open(mask_path)

        # dataset2: masks have 4 channels (RGBA) -> remove alpha channel
        if dataset_id == 2 and mask_img.mode == "RGBA":
            mask_img = mask_img.convert("RGB")
        else:
            mask_img = mask_img.convert("RGB")

        mask_rgb = np.array(mask_img)

        # convert color to class using pre-built lookup table
        color_map = self.color_map1 if dataset_id == 1 else self.color_map2
        lut       = self.lut1       if dataset_id == 1 else self.lut2
        class_mask = mask_rgb_to_class(mask_rgb, color_map, self.num_classes, lut=lut)

        # data augmentation (only training)
        if self.augment:
            image, class_mask = self._augment(image, class_mask)

        # final transformations
        image_tensor = self.img_transform(image)

        # resize mask to image_size with nearest interpolation
        mask_pil = Image.fromarray(class_mask.astype(np.uint8))
        mask_pil = mask_pil.resize(
            (self.image_size, self.image_size),
            resample=Image.NEAREST
        )
        mask_tensor = torch.from_numpy(np.array(mask_pil)).long()

        return image_tensor, mask_tensor

    def _augment(self, image: Image.Image, mask: np.ndarray):
        """random data augmentation to prevent overfitting."""
        # convert mask to pil for synchronized operations
        mask_pil = Image.fromarray(mask.astype(np.uint8))

        # random horizontal flip
        if torch.rand(1).item() > 0.5:
            image = TF.hflip(image)
            mask_pil = TF.hflip(mask_pil)

        # random brightness and contrast (only on image)
        if torch.rand(1).item() > 0.5:
            image = TF.adjust_brightness(image, brightness_factor=np.random.uniform(0.8, 1.2))
        if torch.rand(1).item() > 0.5:
            image = TF.adjust_contrast(image, contrast_factor=np.random.uniform(0.8, 1.2))

        return image, np.array(mask_pil)


# main function to build dataloaders

def build_dataloaders(config_path: str):
    """
    builds training and validation dataloaders using config file.
    supports quick_test mode for debugging (uses small subset of data).
    """
    cfg = load_config(config_path)

    # read color mappings from config.yaml for both datasets
    color_map1 = parse_color_map(cfg["dataset1_color_to_unified"])
    color_map2 = parse_color_map(cfg["dataset2_color_to_unified"])
    
    num_classes = cfg["classes"]["num_classes"]
    image_size  = cfg["data"]["image_size"]
    val_split   = cfg["data"]["val_split"]
    num_workers = cfg["data"]["num_workers"]
    batch_size  = cfg["training"]["batch_size"]
    
    # check if quick test mode is enabled
    quick_test = cfg["data"].get("quick_test", {})
    is_quick_test = quick_test.get("enabled", False)

    # paths are relative to project root (where homework_code folder is)
    config_dir = Path(config_path).parent.parent  # project root

    ds1_imgs  = str(config_dir / cfg["data"]["dataset1_images"])
    ds1_masks = str(config_dir / cfg["data"]["dataset1_masks"])
    ds2_dir   = str(config_dir / cfg["data"]["dataset2_dir"])

    # collect image-mask pairs
    pairs1 = collect_dataset1_pairs(ds1_imgs, ds1_masks)
    pairs2 = collect_dataset2_pairs(ds2_dir)
    all_pairs = pairs1 + pairs2

    print(f"[data] Dataset1: {len(pairs1)} pairs")
    print(f"[data] Dataset2: {len(pairs2)} pairs")
    print(f"[data] Total: {len(all_pairs)} pairs")

    if len(all_pairs) == 0:
        raise RuntimeError("No image-mask pairs found! Check paths in config.yaml.")

    # split into training and validation
    # ensure both datasets are in validation using stratification
    labels = [p[2] for p in all_pairs]   # 1 or 2
    train_pairs, val_pairs = train_test_split(
        all_pairs,
        test_size=val_split,
        random_state=42,
        stratify=labels
    )

    print(f"[data] Train: {len(train_pairs)} | Val: {len(val_pairs)}")
    
    # quick test mode: use only small subset
    if is_quick_test:
        num_train = quick_test.get("num_train_samples", 32)
        num_val = quick_test.get("num_val_samples", 8)
        
        train_pairs = train_pairs[:num_train]
        val_pairs = val_pairs[:num_val]
        
        print(f"[data] QUICK TEST MODE ENABLED")
        print(f"[data] Using only {len(train_pairs)} train samples")
        print(f"[data] Using only {len(val_pairs)} val samples")

    # create datasets
    train_ds = FootballSegDataset(
        train_pairs, color_map1, color_map2,
        num_classes, image_size, augment=True
    )
    val_ds = FootballSegDataset(
        val_pairs, color_map1, color_map2,
        num_classes, image_size, augment=False
    )

    # pin_memory only works with GPU
    pin = torch.cuda.is_available()

    # create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return {
        "train": train_loader,
        "val":   val_loader,
        "num_classes": num_classes,
        "val_pairs": val_pairs,
        "color_map1": color_map1,
        "color_map2": color_map2,
    }


# quick test (run this file directly)

if __name__ == "__main__":
    import sys

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    print(f"Reading config from: {config_path}")

    try:
        loaders = build_dataloaders(str(config_path))
        train_loader = loaders["train"]

        imgs, masks = next(iter(train_loader))
        print(f"\n[test] Image shape:  {imgs.shape}")
        print(f"[test] Mask shape:   {masks.shape}")
        print(f"[test] Unique classes in batch: {masks.unique().tolist()}")
        print("\nData loading successful.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)