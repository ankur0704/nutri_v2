"""
Inference script for the trained emotion classifier.
Use this to test the model after training.
"""

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load model
MODEL_PATH = 'emotion_model'
LABELS = ['angry', 'anxious', 'happy', 'neutral', 'sad', 'stressed']

def load_model():
    """Load the trained model."""
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    return tokenizer, model, device

def predict_emotion(text, tokenizer, model, device):
    """Predict emotion for a given text."""
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
    
    return {
        'emotion': LABELS[pred_idx],
        'confidence': round(confidence, 3),
        'all_scores': {label: round(probs[0][i].item(), 3) for i, label in enumerate(LABELS)}
    }


if __name__ == '__main__':
    print("🧠 Loading emotion classifier...")
    tokenizer, model, device = load_model()
    print(f"   Device: {device}")
    
    # Test examples
    test_texts = [
        "I feel so happy today, everything is going great!",
        "I'm really stressed about my upcoming exam",
        "I feel sad and lonely",
        "I'm so angry at what happened",
        "I'm worried about the future",
        "Today is just a normal day"
    ]
    
    print("\n📊 Predictions:")
    print("-" * 60)
    for text in test_texts:
        result = predict_emotion(text, tokenizer, model, device)
        print(f"Text: \"{text[:50]}...\"")
        print(f"  → Emotion: {result['emotion']} (confidence: {result['confidence']:.1%})")
        print()
