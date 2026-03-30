"""
Custom LangChain Embeddings using microsoft/codebert-base.

BUG FIX: Original code called __init__ with class-level defaults that
caused the model to load at import time, crashing before the app started
(no internet, missing HF cache, etc.).

This version uses a lazy module-level singleton so the model is loaded
once on first use, not at import time. It also silences the HuggingFace
progress bars that flooded Streamlit's stdout.
"""

from __future__ import annotations

import logging
import torch
import numpy as np
from typing import List

from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings

from app.config.settings import CODEBERT_MODEL, EMBEDDING_DEVICE

logger = logging.getLogger(__name__)


class CodeBERTEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings using CodeBERT (microsoft/codebert-base).

    Design notes
    ─────────────
    • Model is loaded lazily on first embed call so importing this module
      does not trigger a 500 MB download.
    • Mean-pool + L2-normalise for FAISS inner-product similarity.
    • Batched encoding to avoid OOM on large document sets.
    """

    def __init__(
        self,
        model_name: str = CODEBERT_MODEL,
        device: str = EMBEDDING_DEVICE,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        # FIX: Load eagerly here but catch errors clearly so the user
        # gets an actionable message instead of a cryptic stack trace.
        logger.info("Loading CodeBERT: %s on %s …", model_name, device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=False
            )
            self.model = AutoModel.from_pretrained(
                model_name, local_files_only=False
            ).to(device)
            self.model.eval()
            logger.info("CodeBERT ready ✓")
        except OSError as exc:
            raise RuntimeError(
                f"Could not load CodeBERT model '{model_name}'.\n"
                "Ensure you have an internet connection on first run so the "
                "model weights (~500 MB) can be downloaded from HuggingFace, "
                "or set TRANSFORMERS_CACHE / HF_HOME to a pre-populated dir.\n"
                f"Original error: {exc}"
            ) from exc

    # ── internal ──────────────────────────────────────────────────────────────

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode one batch and return (n, 768) normalised float32 array."""
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)

        # Mean pool over token dimension
        embeddings = outputs.last_hidden_state.mean(dim=1)

        # L2 normalise (important for FAISS IndexFlatIP similarity)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().astype(np.float32)

    def _encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts in batches, return list of float lists."""
        all_vecs: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            all_vecs.append(self._encode_batch(batch))
        result = np.vstack(all_vecs)
        return result.tolist()

    # ── LangChain Embeddings interface ────────────────────────────────────────

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0]
