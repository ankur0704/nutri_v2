# 📚 ML Concepts Study Guide
## For Custom Emotion Classifier Project

---

# Part 1: Foundations

## 1.1 What is Machine Learning?
Machine Learning is teaching computers to learn patterns from data instead of explicit programming.

**Types:**
- **Supervised Learning**: Learn from labeled data (our emotion classifier)
- **Unsupervised Learning**: Find patterns in unlabeled data
- **Reinforcement Learning**: Learn through trial and reward

---

## 1.2 Neural Networks Basics

### What is a Neuron?
```
Inputs (x1, x2, x3)
        │
        ▼
┌─────────────────────┐
│  y = f(w1*x1 +      │
│       w2*x2 +       │  → Activation → Output
│       w3*x3 + b)    │         f
└─────────────────────┘

Where:
- x = inputs
- w = weights (learnable)
- b = bias (learnable)
- f = activation function
```

### Layers
```
Input Layer → Hidden Layer(s) → Output Layer
    (x)           (features)      (prediction)
```

---

## 1.3 Activation Functions

### ReLU (Rectified Linear Unit)
```
f(x) = max(0, x)

      │
    3 │      ╱
    2 │     ╱
    1 │    ╱
    0 │───╱
   -1 │
      └────────────
       -2 -1 0 1 2 3
```
**Use**: Hidden layers (introduces non-linearity)

### Softmax
```
softmax(xi) = exp(xi) / Σ exp(xj)

Example:
Logits:     [2.0, 1.0, 0.5]
Softmax:    [0.59, 0.24, 0.17]  (sums to 1.0)
```
**Use**: Output layer for multi-class classification

---

## 1.4 Loss Functions

### Cross-Entropy Loss (Classification)
```
Loss = -Σ y_true * log(y_pred)

Example:
True label:    [0, 0, 1, 0, 0, 0]  (sad)
Predicted:     [0.1, 0.1, 0.6, 0.1, 0.05, 0.05]
Loss = -log(0.6) = 0.51
```
**Goal**: Minimize loss (lower = better predictions)

---

## 1.5 Optimizers

### Gradient Descent
```
new_weight = old_weight - learning_rate × gradient

Analogy: Walking downhill to find the lowest point
- Gradient = slope direction
- Learning rate = step size
```

### Adam Optimizer
```
Combines:
1. Momentum (remembers past gradients)
2. RMSprop (adapts learning rate per parameter)

Benefits:
- Faster convergence
- Works well with default settings
```

### AdamW
```
Adam + Weight Decay (regularization)

Prevents overfitting by penalizing large weights
```

---

# Part 2: Natural Language Processing (NLP)

## 2.1 Tokenization

Converting text to numbers that models understand.

```
Input: "I feel happy"

Step 1: Split into tokens
["I", "feel", "happy"]

Step 2: Convert to IDs (using vocabulary)
[1045, 2514, 3407]

Step 3: Add special tokens
[101, 1045, 2514, 3407, 102]
 ↑                       ↑
[CLS]                  [SEP]
```

### Special Tokens
| Token | ID | Purpose |
|-------|-----|---------|
| [CLS] | 101 | Start of sequence, used for classification |
| [SEP] | 102 | End of sequence / separator |
| [PAD] | 0 | Padding for fixed length |
| [UNK] | 100 | Unknown words |

---

## 2.2 Word Embeddings

Words as vectors in multi-dimensional space.

```
"king"  → [0.2, 0.5, 0.8, ...]
"queen" → [0.3, 0.5, 0.7, ...]
"man"   → [0.1, 0.2, 0.3, ...]
"woman" → [0.2, 0.2, 0.2, ...]

Magic property:
king - man + woman ≈ queen
```

**DistilBERT Embedding Size**: 768 dimensions

---

## 2.3 Padding & Truncation

Making all sequences the same length.

```
Max Length: 5

"I feel happy"
Tokens: [101, 1045, 2514, 3407, 102]  ✓ Perfect

"I am very happy today"
Tokens: [101, 1045, 2572, 2200, 3407]  (truncated, [SEP] added)

"Hi"
Tokens: [101, 7632, 102, 0, 0]  (padded with 0s)
```

---

# Part 3: Transformers

## 3.1 The Attention Mechanism

**Core Idea**: Let the model focus on relevant parts of input.

```
Query (Q): "What am I looking for?"
Key (K): "What do I contain?"
Value (V): "What information do I provide?"

Attention = softmax(Q × K^T / √d) × V
```

### Self-Attention Example
```
Input: "The cat sat on the mat"

When processing "sat":
- High attention to "cat" (who sat?)
- High attention to "mat" (where sat?)
- Low attention to "the" (not informative)
```

---

## 3.2 Multi-Head Attention

Multiple attention "heads" looking at different aspects.

```
┌──────────────────────────────────────┐
│            Input Sequence            │
└──────────────────────────────────────┘
        │           │           │
        ▼           ▼           ▼
   ┌────────┐  ┌────────┐  ┌────────┐
   │ Head 1 │  │ Head 2 │  │ Head 3 │
   │Grammar │  │Meaning │  │Emotion │
   └────────┘  └────────┘  └────────┘
        │           │           │
        └─────────┬─┴───────────┘
                  ▼
              Concatenate
                  ▼
           Linear Transform
```

**DistilBERT**: 12 attention heads

---

## 3.3 Transformer Encoder Block

```
┌─────────────────────────────────┐
│         Input Embeddings        │
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│     Multi-Head Self-Attention   │
└─────────────────────────────────┘
              │
              ├──────────────────────┐
              │                      │ (Residual)
              ▼                      │
┌─────────────────────────────────┐ │
│         Add & Normalize         │◄┘
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│      Feed-Forward Network       │
│    (Linear → ReLU → Linear)     │
└─────────────────────────────────┘
              │
              ├──────────────────────┐
              │                      │ (Residual)
              ▼                      │
┌─────────────────────────────────┐ │
│         Add & Normalize         │◄┘
└─────────────────────────────────┘
              │
              ▼
           Output
```

**DistilBERT**: 6 transformer blocks

---

## 3.4 BERT vs DistilBERT

| Feature | BERT | DistilBERT |
|---------|------|------------|
| Layers | 12 | 6 |
| Parameters | 110M | 66M |
| Speed | 1x | 1.6x faster |
| Size | 440MB | 250MB |
| Performance | 100% | 97% |

**DistilBERT** = Knowledge Distillation from BERT
(Student learns from teacher)

---

# Part 4: Transfer Learning

## 4.1 Pre-training

Training on massive unlabeled data.

```
BERT Pre-training Tasks:

1. Masked Language Model (MLM)
   Input: "The [MASK] sat on the mat"
   Predict: "cat"

2. Next Sentence Prediction (NSP)
   Sentence A: "I went to the store."
   Sentence B: "I bought some milk."
   Predict: True (B follows A)
```

**Pre-training Data**: Wikipedia + BookCorpus (3B+ words)

---

## 4.2 Fine-tuning

Adapting pre-trained model to specific task.

```
Pre-trained DistilBERT (General Language Understanding)
              │
              ▼
      Add Classification Head
              │
              ▼
    Train on Emotion Dataset (4,926 samples)
              │
              ▼
Fine-tuned Emotion Classifier (Task-Specific)
```

### Why It Works
```
Pre-trained Knowledge:
- Grammar, syntax
- Word meanings
- Context understanding

+ Fine-tuned Knowledge:
- Emotion-specific patterns
- "feel happy" → happy
- "worried about" → anxious
```

---

## 4.3 Classification Head

```
DistilBERT Output (768 dimensions)
         │
         ▼
┌─────────────────────┐
│ Pre-classifier      │
│ Linear(768 → 768)   │
│ + ReLU              │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Dropout (0.2)       │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Classifier          │
│ Linear(768 → 6)     │
└─────────────────────┘
         │
         ▼
    6 emotion logits
```

---

# Part 5: Training Techniques

## 5.1 Learning Rate Scheduling

```
Learning Rate
    │
2e-5│           ╭────────────────╮
    │          ╱                  ╲
    │         ╱                    ╲
    │        ╱                      ╲
  0 │───────╱                        ╲───
    └────────────────────────────────────
         Warmup    Training       Decay
         (10%)      (80%)         (10%)

Warmup: Gradually increase LR (prevents large initial updates)
Decay: Gradually decrease LR (fine-tune details)
```

---

## 5.2 Gradient Accumulation

Simulating larger batch sizes with limited memory.

```
GPU Memory: 4GB (can fit batch_size=8)
Desired effective batch: 32

Solution:
┌─────────────────────────────────────┐
│ Step 1: Forward + Backward (8)      │
│         Store gradients             │
├─────────────────────────────────────┤
│ Step 2: Forward + Backward (8)      │
│         Add to gradients            │
├─────────────────────────────────────┤
│ Step 3: Forward + Backward (8)      │
│         Add to gradients            │
├─────────────────────────────────────┤
│ Step 4: Forward + Backward (8)      │
│         Add to gradients            │
│         → Update weights!           │
└─────────────────────────────────────┘

8 × 4 = 32 effective batch size
```

---

## 5.3 Class Balancing

Preventing bias towards majority class.

```
Before Balancing:
neutral:  17,099 (50%)  ████████████████████
happy:     7,819 (23%)  ████████████
stressed:    821 (2%)   █

Problem: Model will predict "neutral" too often

After Balancing (821 each):
neutral:    821 (16.7%)  ████████
happy:      821 (16.7%)  ████████
stressed:   821 (16.7%)  ████████

Solution: Undersample majority classes
```

---

## 5.4 Train/Validation/Test Split

```
Total Data: 4,926 samples

┌──────────────────────────────────────────────┐
│                   Train (80%)                 │
│                  3,940 samples                │
│            Used to update weights             │
├────────────────────┬─────────────────────────┤
│   Validation (10%) │      Test (10%)         │
│    493 samples     │     493 samples         │
│   Tune hyperparams │   Final evaluation      │
│   (during training)│   (only once at end)    │
└────────────────────┴─────────────────────────┘
```

---

# Part 6: Evaluation Metrics

## 6.1 Confusion Matrix

```
                    Predicted
                happy  sad  angry
        happy    85    5     10
Actual  sad       3   87     10
        angry     8    4     88

Diagonal = correct predictions
Off-diagonal = errors
```

---

## 6.2 Precision, Recall, F1

```
Precision = TP / (TP + FP)
"Of all predicted positive, how many are actually positive?"

Recall = TP / (TP + FN)
"Of all actual positive, how many did we catch?"

F1 = 2 × (Precision × Recall) / (Precision + Recall)
"Harmonic mean of Precision and Recall"

Example for "happy" class:
- Predicted 100 as happy
- 85 were actually happy (TP=85)
- 15 were other emotions (FP=15)
- 5 happy samples predicted as other (FN=5)

Precision = 85/100 = 85%
Recall = 85/90 = 94%
F1 = 2 × (0.85 × 0.94) / (0.85 + 0.94) = 89%
```

---

## 6.3 Accuracy vs F1

| Metric | When to Use |
|--------|-------------|
| **Accuracy** | Balanced classes |
| **F1-Score** | Imbalanced classes |

```
Example with imbalanced data:
90 spam, 10 ham

Model predicts everything as "spam"
Accuracy = 90% (misleading!)
F1 for ham = 0% (reveals the problem)
```

---

# Part 7: Model Deployment

## 7.1 Saving Models

```python
# Save
model.save_pretrained('emotion_model')
tokenizer.save_pretrained('emotion_model')

# Creates:
emotion_model/
├── config.json          # Model configuration
├── model.safetensors    # Weights (~250MB)
├── tokenizer.json       # Tokenizer config
└── vocab.txt            # Vocabulary
```

---

## 7.2 Inference Pipeline

```python
def predict(text):
    # 1. Tokenize
    inputs = tokenizer(text, return_tensors='pt')
    
    # 2. Forward pass (no gradients needed)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 3. Get probabilities
    probs = softmax(outputs.logits)
    
    # 4. Get prediction
    emotion = labels[argmax(probs)]
    confidence = max(probs)
    
    return emotion, confidence
```

---

## 7.3 CPU vs GPU Inference

| Device | Latency | When to Use |
|--------|---------|-------------|
| GPU | 10-20ms | Batch processing |
| CPU | 50-100ms | Single requests, deployment |

```python
# Check available device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
```

---

# Quick Reference Card

## Key Numbers
```
Model: DistilBERT
Parameters: 66 million
Embedding size: 768
Attention heads: 12
Transformer layers: 6
Vocabulary size: 30,522
Max sequence length: 512 (we use 128)
```

## Training Config
```python
batch_size = 8
gradient_accumulation = 4  # Effective: 32
learning_rate = 2e-5
epochs = 3
warmup_ratio = 0.1
max_length = 128
```

## Emotion Labels
```python
LABELS = ['angry', 'anxious', 'happy', 'neutral', 'sad', 'stressed']
```

---

# 📖 Recommended Reading Order

1. **Week 1**: Neural Network basics (3Blue1Brown YouTube)
2. **Week 2**: NLP fundamentals (tokenization, embeddings)
3. **Week 3**: "Attention Is All You Need" paper (focus on encoder)
4. **Week 4**: BERT paper + Hugging Face Transformers course
5. **Week 5**: Fine-tuning tutorials
6. **Week 6**: Your own project code walkthrough

---

**Good luck with your studies! 🎓**
