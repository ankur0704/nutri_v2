"""
Download and preprocess GoEmotions dataset for emotion classification.
Filters to 6 key emotions relevant for NutriMate AI.
"""

import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# GoEmotions simplified has these label indices:
# 0: admiration, 1: amusement, 2: anger, 3: annoyance, 4: approval,
# 5: caring, 6: confusion, 7: curiosity, 8: desire, 9: disappointment,
# 10: disapproval, 11: disgust, 12: embarrassment, 13: excitement, 14: fear,
# 15: gratitude, 16: grief, 17: joy, 18: love, 19: nervousness,
# 20: optimism, 21: pride, 22: realization, 23: relief, 24: remorse,
# 25: sadness, 26: surprise, 27: neutral

# Map to our 6 emotions
LABEL_MAPPING = {
    17: 'happy',      # joy
    25: 'sad',        # sadness
    2: 'angry',       # anger
    14: 'anxious',    # fear
    11: 'stressed',   # disgust (closest to stressed)
    27: 'neutral',    # neutral
    # Also include related emotions for better data
    1: 'happy',       # amusement
    13: 'happy',      # excitement
    18: 'happy',      # love
    9: 'sad',         # disappointment
    16: 'sad',        # grief
    3: 'angry',       # annoyance
    19: 'anxious',    # nervousness
}

def download_and_filter():
    """Download GoEmotions and filter to our 6 emotions."""
    print("📥 Downloading GoEmotions dataset...")
    dataset = load_dataset('google-research-datasets/go_emotions', 'simplified')
    
    print("🔄 Processing dataset...")
    
    # Process all splits
    train_data = []
    
    for split in ['train', 'validation', 'test']:
        for example in dataset[split]:
            text = example['text']
            labels = example['labels']
            
            # Get the first relevant emotion if any
            for label_id in labels:
                if label_id in LABEL_MAPPING:
                    emotion = LABEL_MAPPING[label_id]
                    train_data.append({
                        'text': text,
                        'emotion': emotion
                    })
                    break  # Only take first matching emotion
    
    # Convert to DataFrame
    df = pd.DataFrame(train_data)
    
    print(f"\n📊 Raw data: {len(df)} samples")
    print(f"\n   Emotion distribution (before balancing):")
    print(df['emotion'].value_counts())
    
    # Balance the dataset - sample equal amounts from each class
    min_samples = min(df['emotion'].value_counts().min(), 2000)
    print(f"\n🔄 Balancing to {min_samples} samples per emotion...")
    
    balanced_dfs = []
    for emotion in df['emotion'].unique():
        emotion_df = df[df['emotion'] == emotion]
        sampled = emotion_df.sample(n=min(len(emotion_df), min_samples), random_state=42)
        balanced_dfs.append(sampled)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    print(f"\n📊 Balanced data: {len(balanced_df)} samples")
    print(f"\n   Emotion distribution (after balancing):")
    print(balanced_df['emotion'].value_counts())
    
    # Shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train/val/test
    train_df, temp_df = train_test_split(
        balanced_df, test_size=0.2, random_state=42, stratify=balanced_df['emotion']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['emotion']
    )
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print(f"\n✅ Saved datasets to data/ folder:")
    print(f"   Train: {len(train_df)} samples -> data/train.csv")
    print(f"   Val: {len(val_df)} samples -> data/val.csv")
    print(f"   Test: {len(test_df)} samples -> data/test.csv")
    
    # Show sample
    print(f"\n📝 Sample data:")
    print(train_df.head(5).to_string(index=False))
    
    print("\n🎉 Dataset ready! Run 'python train_emotion_model.py' next.")
    
    return train_df, val_df, test_df

if __name__ == '__main__':
    download_and_filter()
