import os
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# model with embedder
MODEL_NAME = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"

# path directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "labeled_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUTPUT_EMB = os.path.join(OUTPUT_DIR, "embeddings.npy")
OUTPUT_META = os.path.join(OUTPUT_DIR, "metadata.json")

def load_csv_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} labeled samples.")
    return df

def build_context_text(row):
    # dynamically build to our CSV column names
    return (
        f"Posture: {row.get('posture_label', '')}. "
        f"Gaze: {row.get('gaze_label', '')}. "
        f"Hand movement: {row.get('hand_movement_label', '')}. "
        f"Facial expression: {row.get('facial_expression_label', '')}. "
        f"Total score: {row.get('total_points', '')}."
    )

def compute_embeddings(texts):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    with torch.no_grad():
        out = model(**encoded)
    # mean-pool token embeddings
    embeddings = out.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

def main():
    df = load_csv_data(DATA_PATH)
    texts = [build_context_text(row) for _, row in df.iterrows()]
    
    print("Computing embeddings for labeled data...")
    embeddings = compute_embeddings(texts)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(OUTPUT_EMB, embeddings)
    
    # Save metadata (text + original CSV fields) for retrieval
    meta = []
    for i, (_, row) in enumerate(df.iterrows()):
        meta.append({
            "index": i,
            "text": texts[i],
            "posture_label": row.get("posture_label", ""),
            "gaze_label": row.get("gaze_label", ""),
            "hand_movement_label": row.get("hand_movement_label", ""),
            "facial_expression_label": row.get("facial_expression_label", ""),
            "total_points": row.get("total_points", None)
        })
    with open(OUTPUT_META, "w") as f:
        json.dump(meta, f, indent=2)
    
    print("Done. Embeddings shape:", embeddings.shape)
    print("Saved embeddings to:", OUTPUT_EMB)
    print("Saved metadata to:", OUTPUT_META)

if __name__ == "__main__":
    main()
