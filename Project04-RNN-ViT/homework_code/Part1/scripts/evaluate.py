import torch
from tqdm import tqdm
from sklearn.metrics import classification_report

def evaluate(model, dataloader, criterion, device):
    """Evaluate the model on validation set"""
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
    
    return total_loss / len(dataloader), correct / total


def get_classification_report(model, dataloader, dataset, device):
    """Generate classification report for all labels"""
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
    
    # Generate report
    report = classification_report(all_labels, all_preds, target_names=ner_labels)
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    
    return report