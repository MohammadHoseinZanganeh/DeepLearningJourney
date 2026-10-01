import torch
from sklearn.metrics import classification_report
from tqdm import tqdm


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate model on validation set
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        criterion: Loss function
        device: Device (cuda/cpu)
    
    Returns:
        avg_loss: Average loss
        accuracy: Accuracy score
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits.view(-1, model.num_labels), labels.view(-1))
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            mask = attention_mask.bool()
            correct += (preds[mask] == labels[mask]).sum().item()
            total += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def get_classification_report(model, dataloader, dataset, device):
    """
    Generate classification report for all NER labels
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        dataset: Dataset containing label names
        device: Device (cuda/cpu)
    
    Returns:
        report_str: Classification report as string
        report_dict: Classification report as dictionary
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Report"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)
            
            mask = attention_mask.bool()
            all_preds.extend(preds[mask].cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())
    
    # Get NER label names
    ner_labels = dataset['train'].features['ner_tags'].feature.names
    
    # Generate classification report
    report_str = classification_report(all_labels, all_preds, target_names=ner_labels)
    report_dict = classification_report(all_labels, all_preds, target_names=ner_labels, output_dict=True)
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report_str)
    
    return report_str, report_dict
    