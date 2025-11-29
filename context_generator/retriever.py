from __future__ import annotations
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
EMBEDDINGS_PATH = OUTPUT_DIR / "embeddings.npy"
METADATA_PATH = OUTPUT_DIR / "metadata.json"


class ContextRetriever:
    def __init__(self, top_k: int = 5):
        # Load embeddings + metadata
        self.embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

        with METADATA_PATH.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        assert len(self.embeddings) == len(self.metadata), (
            f"Embeddings count ({len(self.embeddings)}) does not match "
            f"metadata count ({len(self.metadata)})"
        )

        # Normalize embeddings once for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings_norm = self.embeddings / np.clip(norms, 1e-8, None)

        # Load model/tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()

        self.top_k = top_k

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """
        Produce mean-pooled embedding exactly matching how gen_embeddings.py did it.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(self.device)

        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs.attention_mask.unsqueeze(-1)

        masked = hidden * mask
        embedding = masked.sum(1) / mask.sum(1)
        return embedding.cpu().numpy()[0]

    def build_context_from_text(self, query_text: str) -> Dict[str, Any]:
        """
        Embed the query text → retrieve top-K neighbors → compute scores.
        """
        query_vec = self.embed_text(query_text)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        # cosine similarity
        sims = self.embeddings_norm @ query_vec  # [N]
        top_idx = np.argsort(-sims)[: self.top_k]

        neighbors: List[Dict[str, Any]] = []
        total_scores = []

        for idx in top_idx:
            meta = self.metadata[idx]
            sim = float(sims[idx])

            neighbors.append(
                {
                    "description": meta.get("text", ""),
                    "total_points": meta.get("total_points", None),
                    "similarity": sim,
                    "posture": meta.get("posture"),
                    "gaze": meta.get("gaze"),
                    "hand_movement": meta.get("hand_movement"),
                    "facial_expression": meta.get("facial_expression"),
                }
            )

            if isinstance(meta.get("total_points"), (int, float)):
                total_scores.append(meta["total_points"])

        # Aggregate numeric score
        overall_score = float(np.mean(total_scores)) if total_scores else 75.0

        scores = {
            "overall": overall_score,
            "posture": None,
            "gaze": None,
            "gestures": None,
            "facial_expression": None,
        }

        return {
            "scores": scores,
            "neighbors": neighbors,
        }
