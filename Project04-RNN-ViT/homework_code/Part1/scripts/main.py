import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from datetime import datetime

from homework_code.Part1.data.data_loader import load_data, Vocab, create_dataloaders
from homework_code.Part1.models.model import Encoder
from homework_code.Part1.scripts.train import train_model
from homework_code.Part1.utils.visualization import plot_results
from homework_code.Part1.utils.metrics import get_classification_report
from homework_code.Part1.utils.table_visualization import save_report_as_image, save_report_as_text


def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_model(model, model_name, save_dir="models/saved_models"):
    """
    Save model weights
    
    Args:
        model: PyTorch model
        model_name: Name of the model (e.g., 'rnn', 'bilstm', 'bigru')
        save_dir: Directory to save model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/{model_name}_{timestamp}.pth"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'timestamp': timestamp
    }, filename)
    
    print(f" Model saved to: {filename}")
    return filename


def main():
    # 1. Load config
    print("="*50)
    print("LOADING CONFIGURATION")
    print("="*50)
    config = load_config()
    
    # Convert learning_rate to float if it's a string
    if isinstance(config['training']['learning_rate'], str):
        config['training']['learning_rate'] = float(config['training']['learning_rate'])
    
    model_name = config['model']['rnn_type']
    print(f"Model type: {model_name}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    
    # 2. Device
    device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_cuda'] else 'cpu')
    print(f"Using device: {device}")
    
    # 3. Load data
    print("\n" + "="*50)
    print("LOADING DATASET")
    print("="*50)
    dataset = load_data()
    
    # 4. Build vocab
    print("\n" + "="*50)
    print("BUILDING VOCABULARY")
    print("="*50)
    vocab = Vocab(dataset['train'], cutoff=config['training']['cutoff'])
    
    # 5. Create dataloaders
    print("\n" + "="*50)
    print("CREATING DATALOADERS")
    print("="*50)
    train_loader, val_loader = create_dataloaders(
        dataset, vocab, 
        batch_size=config['training']['batch_size']
    )
    
    # Show batch shape
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch['input_ids'].shape}")
    
    # 6. Create model
    print("\n" + "="*50)
    print(f"CREATING MODEL ({model_name.upper()})")
    print("="*50)
    model = Encoder(
        vocab_size=vocab.get_vocab_size(),
        embedding_size=config['model']['embedding_size'],
        hidden_size=config['model']['hidden_size'],
        num_labels=config['model']['num_labels'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        rnn_type=model_name
    )
    model = model.to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 7. Train model
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    history = train_model(
        model, train_loader, val_loader, device,
        num_epochs=config['training']['num_epochs'],
        lr=config['training']['learning_rate']
    )
    
    # 8. Plot results
    print("\n" + "="*50)
    print("PLOTTING RESULTS")
    print("="*50)

    # Create results directory if not exists
    os.makedirs("results", exist_ok=True)

    # Save plot to results folder
    plot_results(history, save_path=f"results/training_results_{model_name}.png")

    # 9. Classification report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    report_str, report_dict = get_classification_report(model, val_loader, dataset, device)
    
    # 10. Save classification report as image and text
    print("\n" + "="*50)
    print("SAVING CLASSIFICATION REPORT")
    print("="*50)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Save as image
    save_report_as_image(report_dict, model_name, save_path="results/")
    
    # Save as text
    save_report_as_text(report_str, model_name, save_path="results/")
    
    # 11. Save model weights
    print("\n" + "="*50)
    print("SAVING MODEL")
    print("="*50)
    save_model(model, model_name)
    
    print("\n" + "="*50)
    print(" TRAINING COMPLETE!")
    print("="*50)


if __name__ == "__main__":
    main()