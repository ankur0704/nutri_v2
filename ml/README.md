# NutriMate AI - Emotion Classifier
Custom emotion classification using DistilBERT.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
```bash
python download_dataset.py
```

### 3. Train model (~30-60 min on GPU)
```bash
python train_emotion_model.py
```

### 4. Test predictions
```bash
python predict_emotion.py
```

## Files
- `download_dataset.py` - Downloads and preprocesses GoEmotions
- `train_emotion_model.py` - Trains DistilBERT (optimized for GTX 1050)
- `predict_emotion.py` - Tests the trained model
- `emotion_model/` - Saved model weights (after training)
