"""
Train DistilBERT for emotion classification.
Optimized for GTX 1050 (4GB VRAM) with gradient accumulation.
"""

import os
import torch
from torch.optim import AdamW
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Configuration - Optimized for GTX 1050 (4GB VRAM)
CONFIG = {
    'model_name': 'distilbert-base-uncased',
    'max_length': 128,          # Reduced for memory
    'batch_size': 8,            # Small batch for 4GB VRAM
    'gradient_accumulation': 4,  # Effective batch = 32
    'epochs': 3,
    'learning_rate': 2e-5,
    'warmup_ratio': 0.1,
    'output_dir': 'emotion_model'
}

# Emotion labels
LABELS = ['angry', 'anxious', 'happy', 'neutral', 'sad', 'stressed']
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}


class EmotionDataset(Dataset):
    """Custom dataset for emotion classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(LABEL2ID[label], dtype=torch.long)
        }


def load_data():
    """Load preprocessed data."""
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')
    return train_df, val_df


def train():
    """Main training function."""
    print("🚀 Starting emotion classifier training...")
    print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer and model
    print("\n📦 Loading DistilBERT...")
    tokenizer = DistilBertTokenizer.from_pretrained(CONFIG['model_name'])
    model = DistilBertForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    model.to(device)
    
    # Load data
    print("📂 Loading data...")
    train_df, val_df = load_data()
    
    train_dataset = EmotionDataset(
        train_df['text'].tolist(),
        train_df['emotion'].tolist(),
        tokenizer,
        CONFIG['max_length']
    )
    
    val_dataset = EmotionDataset(
        val_df['text'].tolist(),
        val_df['emotion'].tolist(),
        tokenizer,
        CONFIG['max_length']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'] * 2)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    total_steps = len(train_loader) * CONFIG['epochs'] // CONFIG['gradient_accumulation']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * CONFIG['warmup_ratio']),
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"\n🏋️ Training for {CONFIG['epochs']} epochs...")
    best_accuracy = 0
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for i, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / CONFIG['gradient_accumulation']
            loss.backward()
            total_loss += loss.item()
            
            if (i + 1) % CONFIG['gradient_accumulation'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            pbar.set_postfix({'loss': f'{loss.item() * CONFIG["gradient_accumulation"]:.4f}'})
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"\n📊 Epoch {epoch+1} - Val Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Save best model
            os.makedirs(CONFIG['output_dir'], exist_ok=True)
            model.save_pretrained(CONFIG['output_dir'])
            tokenizer.save_pretrained(CONFIG['output_dir'])
            print(f"   ✅ Saved best model (accuracy: {accuracy:.4f})")
    
    print(f"\n🎉 Training complete! Best accuracy: {best_accuracy:.4f}")
    print(f"   Model saved to: {CONFIG['output_dir']}/")
    
    # Final evaluation
    print("\n📋 Final Classification Report:")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=LABELS
    ))


if __name__ == '__main__':
    train()
