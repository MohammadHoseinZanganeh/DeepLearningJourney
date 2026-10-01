import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import numpy as np
import yaml
from data.data_loader import get_data_loaders
from models.model import VisionTransformer

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def visualize_attention():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    _, _, test_loader = get_data_loaders(
        batch_size=2,
        num_workers=0,
        subset_fraction=1.0,
        resize=config['data'].get('resize', None)
    )
    
    model_config = config['model']
    model = VisionTransformer(
        image_size=model_config['image_size'],
        patch_size=model_config['patch_size'],
        in_channels=model_config['in_channels'],
        num_classes=model_config['num_classes'],
        embed_dim=model_config['embed_dim'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        mlp_ratio=model_config['mlp_ratio'],
        dropout=model_config.get('dropout', 0.0),
        attention_dropout=model_config.get('attention_dropout', 0.0)
    )
    
    model_path = "models/saved_models/best_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded from best_model.pth")
    else:
        print("Warning: model file not found. Using random weights.")
    
    model = model.to(device)
    model.eval()
    
    images, labels = next(iter(test_loader))
    images = images[:2].to(device)
    
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(device)
    
    def denormalize(img):
        return (img * std + mean).clamp(0, 1)
    
    def modified_forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn
    
    original_forwards = []
    target_layers = [0, 1, 3]
    for idx in target_layers:
        attn_module = model.blocks[idx].attn
        original_forwards.append(attn_module.forward)
        attn_module.forward = modified_forward.__get__(attn_module, type(attn_module))
    
    with torch.no_grad():
        x = model.patch_embed(images)
        cls_tokens = model.cls_token.expand(images.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + model.pos_embed
        x = model.pos_drop(x)
        
        attention_maps = []
        for i, blk in enumerate(model.blocks):
            if i in target_layers:
                norm1_out = blk.norm1(x)
                attn_out, attn_weights = blk.attn(norm1_out)
                x = x + attn_out
                norm2_out = blk.norm2(x)
                mlp_out = blk.mlp(norm2_out)
                x = x + mlp_out
                attention_maps.append(attn_weights)
            else:
                x = blk(x)
        
        for idx, orig_forward in zip(target_layers, original_forwards):
            model.blocks[idx].attn.forward = orig_forward
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    os.makedirs("results/attention", exist_ok=True)
    
    patch_size = model_config['patch_size']
    grid_size = model_config['image_size'] // patch_size
    
    for img_idx in range(2):
        img_denorm = denormalize(images[img_idx])
        img_disp = img_denorm.squeeze().cpu().permute(1, 2, 0).numpy()
        
        # Original image (no changes)
        original_img = img_disp.copy()
        h, w = original_img.shape[0], original_img.shape[1]
        step_h, step_w = h // grid_size, w // grid_size
        
        fig = plt.figure(figsize=(12, 4 * len(target_layers)))
        gs = fig.add_gridspec(len(target_layers), 4, width_ratios=[0.3, 1, 1, 1], 
                              wspace=0.05, hspace=0.05)
        
        for row, layer_idx in enumerate(target_layers):
            attn = attention_maps[row]
            cls_attn = attn[img_idx, :, 0, 1:]
            
            # Column 0: L1, L2, L4 label
            ax_label = fig.add_subplot(gs[row, 0])
            ax_label.axis('off')
            ax_label.text(0.5, 0.5, f"L{layer_idx+1}", fontsize=22, weight='bold',
                         horizontalalignment='center', verticalalignment='center')
            
            # Column 1: Patched Input with grid lines
            ax_patch = fig.add_subplot(gs[row, 1])
            ax_patch.imshow(original_img, aspect='equal', extent=[0, w, h, 0])
            
            # Draw grid lines using exact pixel coordinates
            # Horizontal lines (y coordinates)
            for i in range(1, grid_size):
                ax_patch.axhline(y=i * step_h, color='white', linewidth=0.5, alpha=0.7)
            # Vertical lines (x coordinates)
            for i in range(1, grid_size):
                ax_patch.axvline(x=i * step_w, color='white', linewidth=0.5, alpha=0.7)
            
            # Set xlim and ylim to match image exactly
            ax_patch.set_xlim(0, w)
            ax_patch.set_ylim(h, 0)
            ax_patch.axis('off')
            
            # Column 2: Head 1
            head_attn = cls_attn[0].reshape(grid_size, grid_size).cpu().numpy()
            ax_head = fig.add_subplot(gs[row, 2])
            ax_head.imshow(head_attn, cmap='hot', aspect='equal')
            ax_head.axis('off')
            
            # Column 3: Mean Heads
            mean_attn = cls_attn.mean(dim=0).reshape(grid_size, grid_size).cpu().numpy()
            ax_mean = fig.add_subplot(gs[row, 3])
            ax_mean.imshow(mean_attn, cmap='hot', aspect='equal')
            ax_mean.axis('off')
        
        # Add titles
        title_positions = [1, 2, 3]
        titles = ["Patched Input", "First Attention Head", "Mean Attention of Heads"]
        
        for pos, title in zip(title_positions, titles):
            ax_title = fig.add_subplot(gs[0, pos])
            ax_title.axis('off')
            ax_title.set_title(title, fontsize=13, pad=10)
        
        plt.suptitle(f"Image {img_idx+1} - True Label: {class_names[labels[img_idx].item()]}", 
                     fontsize=14, y=1.02)
        plt.tight_layout()
        
        save_path = f"results/attention/img{img_idx+1}_attention.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image {img_idx+1} saved in {save_path}")
        plt.show()
        plt.close()
    
    print("All attention maps saved in results/attention/")

if __name__ == "__main__":
    visualize_attention()