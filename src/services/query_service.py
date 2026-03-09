import logging

from src.core.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class QueryService:
    """Handles query processing through the RAG pipeline."""

    def __init__(self, pipeline: RAGPipeline) -> None:
        self._pipeline = pipeline

    def ask(self, question: str) -> dict:
        """Process a user question through the RAG pipeline."""
        logger.info("Processing query: %s", question[:100])
        result = self._pipeline.query(question)
        logger.info("Query answered. Sources: %d", len(result.get("sources", [])))
        return result
