from sentence_transformers import SentenceTransformer

from src.core.interfaces.embedder import Embedder


class SentenceTransformerEmbedder(Embedder):
    """Embedder backed by HuggingFace sentence-transformers (fully open-source)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu") -> None:
        self._model = SentenceTransformer(model_name, device=device)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embedding = self._model.encode([query], show_progress_bar=False)
        return embedding[0].tolist()
