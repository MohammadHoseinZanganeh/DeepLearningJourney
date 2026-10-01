import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward
        logits = model(input_ids, attention_mask)
        loss = criterion(logits.view(-1, model.num_labels), labels.view(-1))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy
        preds = torch.argmax(logits, dim=-1)
        mask = attention_mask.bool()
        correct += (preds[mask] == labels[mask]).sum().item()
        total += mask.sum().item()
    
    return total_loss / len(dataloader), correct / total


def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-3):
    """Full training loop"""
    from homework_code.Part1.scripts.evaluate import evaluate  # import inside to avoid circular
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    return history