import numpy as np
from sentence_transformers import SentenceTransformer

class LocalEmbedder:
    """Local embedding via sentence-transformers (downloads model on first run)."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    @property
    def dim(self) -> int:
        # all-MiniLM-L6-v2 is 384-dim
        return 384

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")
