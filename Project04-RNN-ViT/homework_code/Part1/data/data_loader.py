from datasets import load_dataset, load_from_disk
import os


import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from collections import Counter


DATA_PATH = "data/dataset/conll2003"


def download_and_save():
    dataset = load_dataset("conll2003")
    dataset.save_to_disk(DATA_PATH)
    return dataset


def load_data():
    if os.path.exists(DATA_PATH):
        return load_from_disk(DATA_PATH)
    else:
        return download_and_save()


# VOCAB CLASS 
class Vocab:
    """
    Vocabulary class for token <-> id mapping
    """
    def __init__(self, train_data, cutoff=5):
        self.cutoff = cutoff
        self.word2idx = {}
        self.idx2word = {}
        self.special_tokens = ['<pad>', '<unk>', '<s>', '</s>']
        self.build_vocab(train_data)
    
    def build_vocab(self, train_data):
        """
        Build vocabulary from training data
        Only keep tokens with frequency >= cutoff
        """
        # Collect all tokens
        all_tokens = []
        for example in train_data:
            all_tokens.extend(example['tokens'])
        
        # Count frequencies
        token_counts = Counter(all_tokens)
        
        # Add special tokens first
        for idx, token in enumerate(self.special_tokens):
            self.word2idx[token] = idx
        
        # Add frequent tokens
        current_idx = len(self.special_tokens)
        for token, count in token_counts.items():
            if count >= self.cutoff:
                self.word2idx[token] = current_idx
                current_idx += 1
        
        # Build reverse mapping
        self.idx2word = {idx: token for token, idx in self.word2idx.items()}
        
        print(f" Vocabulary built successfully!")
        print(f"  - Total tokens: {len(all_tokens)}")
        print(f"  - Unique tokens: {len(token_counts)}")
        print(f"  - Vocab size (cutoff={self.cutoff}): {len(self.word2idx)}")
    
    def token_to_id(self, token):
        """Convert token to id, return <unk> if not found"""
        return self.word2idx.get(token, self.word2idx['<unk>'])
    
    def id_to_token(self, idx):
        """Convert id to token, return <unk> if not found"""
        return self.idx2word.get(idx, '<unk>')
    
    def get_vocab_size(self):
        """Return vocabulary size"""
        return len(self.word2idx)


# COLLATE FUNCTION 
def collate_fn(batch, vocab, max_len=None):
    """
    Collate function for DataLoader with padding
    """
    tokens = [example['tokens'] for example in batch]
    ner_tags = [example['ner_tags'] for example in batch]
    
    # Convert tokens to IDs
    token_ids = []
    for sent in tokens:
        ids = [vocab.token_to_id(token) for token in sent]
        token_ids.append(ids)
    
    # Find max length in batch
    batch_max_len = max(len(ids) for ids in token_ids)
    
    # Pad sequences
    padded_ids = []
    padded_labels = []
    attention_masks = []
    
    for i, ids in enumerate(token_ids):
        # Pad tokens
        padded = ids + [0] * (batch_max_len - len(ids))
        padded_ids.append(padded)
        
        # Pad labels with -100 (ignore in loss)
        labels = ner_tags[i] + [-100] * (batch_max_len - len(ner_tags[i]))
        padded_labels.append(labels)
        
        # Attention mask (1 for real, 0 for padding)
        mask = [1] * len(ids) + [0] * (batch_max_len - len(ids))
        attention_masks.append(mask)
    
    return {
        'input_ids': torch.tensor(padded_ids, dtype=torch.long),
        'labels': torch.tensor(padded_labels, dtype=torch.long),
        'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
    }




def create_dataloaders(dataset, vocab, batch_size=32):
    """
    Create train and validation DataLoaders
    """
    # Create DataLoader with collate function
    train_loader = DataLoader(
        dataset['train'],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, vocab)
    )
    
    val_loader = DataLoader(
        dataset['validation'],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, vocab)
    )
    
    return train_loader, val_loader


#  MAIN (for testing)
if __name__ == "__main__":
    # Test the data loader
    dataset = load_data()
    train_data = dataset['train']
    
    # Create vocab
    vocab = Vocab(train_data, cutoff=5)
    
    # Test vocab functions
    print(f"\n=== Testing Vocab ===")
    print(f"Vocab size: {vocab.get_vocab_size()}")
    print(f"Token with id 0: '{vocab.id_to_token(0)}'")
    print(f"ID of 'quick': {vocab.token_to_id('quick')}")
    
    # Test DataLoader
    print(f"\n=== Testing DataLoader ===")
    train_loader, val_loader = create_dataloaders(dataset, vocab, batch_size=32)
    
    # Get first batch
    batch = next(iter(train_loader))
    print(f"Input shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Mask shape: {batch['attention_mask'].shape}")