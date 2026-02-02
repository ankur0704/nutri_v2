# 🧠 Custom Emotion Classifier - Technical Documentation


> 
> This document provides a comprehensive technical explanation of the custom emotion classification model built for NutriMate AI.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Dataset](#dataset)
4. [Training Process](#training-process)
5. [Key Concepts Explained](#key-concepts-explained)
6. [Performance Metrics](#performance-metrics)
7. [Integration & Inference](#integration--inference)
8. [Academic Significance](#academic-significance)

---

## Overview

### What We Built
A **fine-tuned DistilBERT model** that classifies text into 6 emotion categories:

| Emotion | Description | Example Input |
|---------|-------------|---------------|
| **happy** | Positive, joyful | "I feel amazing today!" |
| **sad** | Melancholic, down | "I feel lonely and tired" |
| **angry** | Frustrated, irritated | "This is so unfair!" |
| **anxious** | Worried, fearful | "I'm nervous about my exam" |
| **stressed** | Overwhelmed, pressured | "I have too much work" |
| **neutral** | No strong emotion | "Today is a normal day" |

### Why Custom Model Instead of API?

| Approach | Pros | Cons |
|----------|------|------|
| **Gemini/ChatGPT API** | Easy, no training needed | Costly, internet required, no control |
| **Custom Model (Ours)** | Free, offline, full control, academic value | Requires training, limited to trained classes |

---

## Model Architecture

### Base Model: DistilBERT

```
DistilBERT (Distilled BERT)
├── 6 Transformer Layers (vs 12 in BERT)
├── 66 Million Parameters (vs 110M in BERT)
├── 40% smaller, 60% faster
└── 97% of BERT's performance
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT TEXT                                │
│         "I feel really stressed about my exam"              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     TOKENIZER                                │
│   Converts text to numerical tokens                         │
│   [CLS] I feel really stressed about my exam [SEP]          │
│   [101, 1045, 2514, 2428, 13233, 2055, 2026, 4566, 102]     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   DISTILBERT ENCODER                         │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Embedding Layer (768 dimensions)                    │   │
│   ├─────────────────────────────────────────────────────┤   │
│   │  Transformer Layer 1                                 │   │
│   │    - Multi-Head Self-Attention (12 heads)           │   │
│   │    - Feed-Forward Network                           │   │
│   ├─────────────────────────────────────────────────────┤   │
│   │  Transformer Layer 2-6 (same structure)             │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
│   Output: 768-dimensional vector for [CLS] token            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 CLASSIFICATION HEAD                          │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Pre-classifier: Linear(768 → 768) + ReLU           │   │
│   ├─────────────────────────────────────────────────────┤   │
│   │  Dropout: 0.2 (regularization)                      │   │
│   ├─────────────────────────────────────────────────────┤   │
│   │  Classifier: Linear(768 → 6)                        │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      SOFTMAX                                 │
│   Converts logits to probabilities                          │
│   [0.02, 0.05, 0.03, 0.15, 0.72, 0.03]                     │
│    angry anxious happy neutral stressed sad                  │
│                            ↓                                 │
│              Predicted: STRESSED (72%)                       │
└─────────────────────────────────────────────────────────────┘
```

### Key Components Explained

#### 1. Tokenizer
Converts human-readable text into numerical tokens that the model understands.

```python
# Example
tokenizer("I feel happy")
# Output: [101, 1045, 2514, 3407, 102]
# Where: 101 = [CLS], 102 = [SEP], 2514 = "feel", 3407 = "happy"
```

#### 2. Transformer Layers
Each layer processes the input through:
- **Self-Attention**: Understands relationships between words
- **Feed-Forward Network**: Transforms representations

#### 3. Classification Head
A simple neural network that maps the 768-dimensional representation to 6 emotion classes.

---

## Dataset

### Source: GoEmotions (Google Research)

| Property | Value |
|----------|-------|
| **Original Size** | 58,000 Reddit comments |
| **Original Labels** | 27 fine-grained emotions |
| **Our Filtered Size** | 34,035 samples |
| **Our Balanced Size** | 4,926 samples (821 per class) |

### Label Mapping

We mapped Google's 27 emotions to our 6 categories:

```python
# Original GoEmotions → Our Labels
{
    'joy': 'happy',
    'amusement': 'happy',
    'excitement': 'happy',
    'love': 'happy',
    
    'sadness': 'sad',
    'disappointment': 'sad',
    'grief': 'sad',
    
    'anger': 'angry',
    'annoyance': 'angry',
    
    'fear': 'anxious',
    'nervousness': 'anxious',
    
    'disgust': 'stressed',
    
    'neutral': 'neutral'
}
```

### Data Split

```
Total: 4,926 samples
├── Train: 3,940 (80%)
├── Validation: 493 (10%)
└── Test: 493 (10%)
```

### Why Balancing Matters

**Before Balancing:**
```
neutral:  17,099 (50.2%)  ████████████████████
happy:     7,819 (23.0%)  ████████████
angry:     4,641 (13.6%)  ███████
sad:       2,789 (8.2%)   ████
anxious:     866 (2.5%)   █
stressed:    821 (2.4%)   █
```

**After Balancing (821 each):**
```
neutral:  821 (16.7%)  ████████
happy:    821 (16.7%)  ████████
angry:    821 (16.7%)  ████████
sad:      821 (16.7%)  ████████
anxious:  821 (16.7%)  ████████
stressed: 821 (16.7%)  ████████
```

Without balancing, the model would be biased towards predicting "neutral" (50% of data).

---

## Training Process

### Hyperparameters (Optimized for GTX 1050 4GB)

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| **Batch Size** | 8 | Small to fit in 4GB VRAM |
| **Gradient Accumulation** | 4 | Effective batch = 32 |
| **Learning Rate** | 2e-5 | Standard for fine-tuning BERT |
| **Epochs** | 3 | Prevent overfitting |
| **Max Sequence Length** | 128 | Truncate long texts |
| **Warmup Ratio** | 0.1 | Gradual learning rate increase |

### Training Loop Pseudocode

```python
for epoch in range(3):
    for batch in training_data:
        # 1. Forward pass
        predictions = model(batch.text)
        
        # 2. Calculate loss
        loss = CrossEntropyLoss(predictions, batch.labels)
        
        # 3. Backward pass (gradient accumulation)
        loss.backward()
        
        # 4. Update weights (every 4 batches)
        if step % 4 == 0:
            optimizer.step()
            scheduler.step()
    
    # 5. Validate
    accuracy = evaluate(validation_data)
    
    # 6. Save best model
    if accuracy > best_accuracy:
        save_model()
```

### Loss Function: Cross-Entropy

```
Loss = -Σ y_true * log(y_pred)

Where:
- y_true = actual emotion (one-hot encoded)
- y_pred = predicted probabilities
```

### Optimizer: AdamW

```
AdamW = Adam + Weight Decay

Benefits:
- Adaptive learning rates per parameter
- Momentum for faster convergence
- Weight decay for regularization
```

### Learning Rate Schedule

```
Learning Rate
    │
2e-5│           ╭───────────────────╮
    │          ╱                     ╲
    │         ╱                       ╲
    │        ╱                         ╲
  0 │───────╱                           ╲────
    └──────────────────────────────────────────
        Warmup        Training          Decay
        (10%)          (80%)           (10%)
```

---

## Key Concepts Explained

### 1. Transfer Learning

Instead of training from scratch (which would need millions of samples), we:

1. **Start with pre-trained DistilBERT** (trained on Wikipedia + BookCorpus)
2. **Add our classification head** (6 neurons for 6 emotions)
3. **Fine-tune on our data** (4,926 samples)

```
Pre-trained Knowledge                Our Task-Specific Knowledge
┌────────────────────┐              ┌────────────────────┐
│ - Grammar          │              │ - Emotion words    │
│ - Syntax           │     +        │ - Context clues    │
│ - Word meanings    │              │ - Sentiment        │
│ - General language │              │ - 6 categories     │
└────────────────────┘              └────────────────────┘
         ↓                                   ↓
         └───────────────┬───────────────────┘
                         ↓
              Fine-tuned Emotion Classifier
```

### 2. Self-Attention Mechanism

The core of transformers - allows the model to understand word relationships:

```
Input: "I feel happy because I passed my exam"

Attention Matrix (simplified):
          I   feel  happy  because  I   passed  my   exam
I         ■   ■     ░      ░        ■   ░       ░    ░
feel      ■   ■     ■      ░        ░   ░       ░    ░
happy     ░   ■     ■      ■        ░   ■       ░    ░
because   ░   ░     ■      ■        ░   ░       ░    ░
I         ■   ░     ░      ░        ■   ■       ■    ░
passed    ░   ░     ■      ░        ■   ■       ░    ■
my        ░   ░     ░      ░        ■   ░       ■    ■
exam      ░   ░     ░      ░        ░   ■       ■    ■

■ = high attention (strong relationship)
░ = low attention (weak relationship)

Key insight: "happy" attends to "passed" and "feel" - understands context!
```

### 3. Softmax Function

Converts raw model outputs (logits) to probabilities:

```python
softmax(x_i) = exp(x_i) / Σ exp(x_j)

Example:
Raw logits:    [-1.2, 0.5, 2.1, 0.3, -0.5, 0.8]
After softmax: [0.02, 0.11, 0.56, 0.09, 0.04, 0.15]
                ↑                 ↑
             angry              happy (predicted)

Sum = 1.0 (valid probability distribution)
```

### 4. Gradient Accumulation

With limited GPU memory, we simulate larger batch sizes:

```
Physical Batch Size: 8
Accumulation Steps: 4
Effective Batch Size: 8 × 4 = 32

Process:
Step 1: Forward + Backward on 8 samples (store gradients)
Step 2: Forward + Backward on 8 samples (add to gradients)
Step 3: Forward + Backward on 8 samples (add to gradients)
Step 4: Forward + Backward on 8 samples (add to gradients)
        → Now update weights with accumulated gradients
```

---

## Performance Metrics

### Results on Test Set

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| angry | 0.68 | 0.72 | 0.70 |
| anxious | 0.75 | 0.71 | 0.73 |
| happy | 0.85 | 0.89 | 0.87 |
| neutral | 0.72 | 0.69 | 0.70 |
| sad | 0.78 | 0.76 | 0.77 |
| stressed | 0.65 | 0.68 | 0.66 |

### Metric Definitions

- **Precision**: Of all predicted X, how many were actually X?
  ```
  Precision = True Positives / (True Positives + False Positives)
  ```

- **Recall**: Of all actual X, how many did we correctly predict?
  ```
  Recall = True Positives / (True Positives + False Negatives)
  ```

- **F1-Score**: Harmonic mean of Precision and Recall
  ```
  F1 = 2 × (Precision × Recall) / (Precision + Recall)
  ```

### Confidence Distribution

```
Confidence Level    Percentage of Predictions
90-100%             ████████████ 25%
70-89%              ████████████████████ 45%
50-69%              ████████████ 22%
<50%                ████ 8%
```

---

## Integration & Inference

### Backend Integration

```python
# 1. Load model on server startup
emotion_tokenizer = DistilBertTokenizer.from_pretrained('emotion_model')
emotion_model = DistilBertForSequenceClassification.from_pretrained('emotion_model')

# 2. Inference function
def predict_emotion(text):
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get prediction
    probs = softmax(outputs.logits)
    emotion = LABELS[argmax(probs)]
    confidence = max(probs)
    
    return {"emotion": emotion, "confidence": confidence}
```

### API Endpoint

```bash
POST /analyze_emotion
Content-Type: application/json

{"text": "I feel really happy today"}

Response:
{
    "emotion": "happy",
    "confidence": 0.898,
    "all_scores": {
        "angry": 0.02,
        "anxious": 0.03,
        "happy": 0.898,
        "neutral": 0.02,
        "sad": 0.01,
        "stressed": 0.02
    },
    "model": "custom_distilbert"
}
```

### Inference Time

| Device | Latency |
|--------|---------|
| CPU (i5 9th gen) | ~50-100ms |
| GPU (GTX 1050) | ~10-20ms |

---

## Academic Significance

### What Makes This Unique

1. **Custom-Trained Model**: Not just an API wrapper
2. **Domain-Specific Fine-tuning**: Trained on emotion data relevant to wellness
3. **End-to-End Pipeline**: From data collection to deployment
4. **Reproducible Research**: All code and methods documented

### Comparison with Existing Approaches

| Approach | Accuracy | Latency | Cost | Control |
|----------|----------|---------|------|---------|
| Rule-based (keywords) | 40-50% | <1ms | Free | Full |
| Pre-trained Sentiment | 60-70% | 10ms | Free | Limited |
| **Our Custom Model** | **70-85%** | **50ms** | **Free** | **Full** |
| GPT-4 API | 85-95% | 500ms+ | $0.01/req | None |

### Future Improvements

1. **More Training Data**: Use additional datasets (ISEAR, AffectiveText)
2. **Multi-label Classification**: Detect multiple emotions simultaneously
3. **Explainability (XAI)**: Show which words influenced the prediction
4. **Continual Learning**: Update model as more user data is collected

---

## File Structure

```
ml/
├── requirements.txt        # Python dependencies
├── download_dataset.py     # Dataset preparation script
├── train_emotion_model.py  # Training script
├── predict_emotion.py      # Standalone inference script
├── data/
│   ├── train.csv          # Training data (3,940 samples)
│   ├── val.csv            # Validation data (493 samples)
│   └── test.csv           # Test data (493 samples)
└── emotion_model/
    ├── config.json        # Model configuration
    ├── model.safetensors  # Model weights (~250MB)
    ├── tokenizer.json     # Tokenizer vocabulary
    └── vocab.txt          # Word vocabulary
```

---

## References

1. Sanh, V., et al. (2019). "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"
2. Demszky, D., et al. (2020). "GoEmotions: A Dataset of Fine-Grained Emotions" - Google Research
3. Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
4. Wolf, T., et al. (2020). "Transformers: State-of-the-Art Natural Language Processing" - Hugging Face

---

<p align="center">
  <strong>NutriMate AI - Custom Emotion Classifier</strong><br>
  
</p>
