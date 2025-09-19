"""
Document Processing Module for ResearchGPT Assistant
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import re
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DocumentProcessor:
    """Processes PDF documents for text extraction, chunking, and similarity search.

    Extracts text from PDFs, preprocesses it, chunks it into segments, and builds a TF-IDF search index.
    """

    def __init__(self, config):
        """Initialize the DocumentProcessor with configuration and resources.

        Args:
            config: Configuration object with settings (e.g., CHUNK_SIZE, MIN_CHUNK_SIZE).

        Raises:
            LookupError: If NLTK 'punkt' tokenizer is not available and cannot be downloaded.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_df=0.85,
            min_df=2,
            strip_accents="unicode",
            lowercase=True,
        )
        self.documents: Dict = {}
        self.document_vectors = None
        self.all_chunks: List[str] = []
        self.chunk_doc_ids: List[str] = []
        self.min_chunk_size: int = self.config.MIN_CHUNK_SIZE

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Extracted and preprocessed text.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            extracted_text = "\n\n".join(page.extract_text(layout=True) or "" for page in pdf.pages)
        cleaned_text = self.preprocess_text(extracted_text)
        self.logger.debug(f"Extracted text from {pdf_path}, length: {len(cleaned_text)}")
        return cleaned_text

    def preprocess_text(self, text: str) -> str:
        """Preprocess text by cleaning and normalizing it.

        Removes hyphenated line breaks, URLs, DOIs, arXiv references, special characters,
        and normalizes whitespace.

        Args:
            text: Raw text to preprocess.

        Returns:
            Cleaned and normalized text.
        """
        if not text:
            return ""
        text = re.sub(r"-\n", "", text)  # Remove hyphenated line breaks
        text = re.sub(r"\n\s*\n+", "\n\n", text)  # Normalize newlines
        text = re.sub(r"[ \t]+", " ", text.strip())  # Normalize whitespace
        text = re.sub(r"\b(arXiv|doi|https?://[^\s]+)\b", "", text)  # Remove URLs, DOIs, arXiv
        text = re.sub(r"[^\w\s.,!?()-]", " ", text)  # Remove special characters
        return re.sub(r"\s+", " ", text)  # Final whitespace normalization

    def chunk_text(self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
        """Chunk text into smaller segments based on sentences.

        Args:
            text: Text to chunk.
            chunk_size: Target size for each chunk (default: config.CHUNK_SIZE).
            overlap: Number of characters to overlap between chunks (default: 20% of chunk_size).

        Returns:
            List of text chunks.
        """
        if not text:
            return []
        chunk_size = chunk_size or self.config.CHUNK_SIZE
        overlap = overlap or int(self.config.CHUNK_SIZE * 0.2)
        self.logger.debug(f"Chunking with CHUNK_SIZE={chunk_size}, MIN_CHUNK_SIZE={self.min_chunk_size}")

        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                current_chunk = current_chunk[-overlap:] + " " + sentence if overlap > 0 else sentence
            else:
                current_chunk += " " + sentence

        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())

        self.logger.debug(f"Created {len(chunks)} chunks")
        return chunks

    def process_document(self, pdf_path: str) -> str:
        """Process a PDF document by extracting, preprocessing, chunking, and storing metadata.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Document ID (filename stem).
        """
        pdf_path = Path(pdf_path)
        doc_id = pdf_path.stem
        raw_text = self.extract_text_from_pdf(pdf_path)
        if not raw_text:
            self.logger.warning(f"No text extracted from {doc_id}")
            return doc_id
        cleaned_text = self.preprocess_text(raw_text)
        chunks = self.chunk_text(cleaned_text)
        metadata = {
            "title": doc_id.replace("_", " ").title(),
            "file_path": str(pdf_path),
            "length": len(cleaned_text),
            "num_chunks": len(chunks),
            "created_at": pd.Timestamp.now().isoformat(),
        }
        self.documents[doc_id] = {"title": metadata["title"], "chunks": chunks, "metadata": metadata}
        self.logger.info(f"Processed document {doc_id} with {len(chunks)} chunks")
        return doc_id

    def build_search_index(self) -> None:
        """Build a TF-IDF search index from processed document chunks."""
        if not self.documents:
            self.logger.warning("No documents available to build search index")
            return
        self.all_chunks = [chunk for doc_data in self.documents.values() for chunk in doc_data["chunks"]]
        if not self.all_chunks:
            self.logger.warning("No chunks available to build search index")
            return
        self.chunk_doc_ids = [doc_id for doc_id, doc_data in self.documents.items() for _ in doc_data["chunks"]]
        self.document_vectors = self.vectorizer.fit_transform(self.all_chunks)
        self.logger.info(f"Built TF-IDF index with {len(self.all_chunks)} chunks")

    def find_similar_chunks(self, query: str, top_k: int = 5, min_score: float = 0.1) -> List[Tuple[str, float, str]]:
        """Find text chunks similar to the query using cosine similarity.

        Args:
            query: Search query text.
            top_k: Maximum number of results to return (default: 5).
            min_score: Minimum similarity score for results (default: 0.1).

        Returns:
            List of tuples containing (chunk, similarity_score, document_id).
        """
        if self.document_vectors is None or self.document_vectors.shape[0] == 0:
            return []
        cleaned_query = self.preprocess_text(query)
        if not cleaned_query:
            return []
        query_vector = self.vectorizer.transform([cleaned_query])
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        top_indices = np.argsort(similarities)[::-1][:min(top_k, len(similarities))]
        return [
            (self.all_chunks[idx], float(similarities[idx]), self.chunk_doc_ids[idx])
            for idx in top_indices
            if similarities[idx] > min_score
        ]

    def get_document_stats(self) -> Dict:
        """Calculate statistics for processed documents.

        Returns:
            Dictionary with statistics: num_documents, total_chunks, average_length,
            average_chunk_length, and document_titles.
        """
        if not self.documents:
            return {
                "num_documents": 0,
                "total_chunks": 0,
                "average_length": 0,
                "average_chunk_length": 0,
                "document_titles": [],
            }
        total_chunks = sum(len(doc_data["chunks"]) for doc_data in self.documents.values())
        stats = {
            "num_documents": len(self.documents),
            "total_chunks": total_chunks,
            "average_length": sum(doc_data["metadata"]["length"] for doc_data in self.documents.values())
            / len(self.documents),
            "average_chunk_length": (
                sum(len(chunk) for doc_data in self.documents.values() for chunk in doc_data["chunks"])
                / total_chunks
                if total_chunks > 0
                else 0
            ),
            "document_titles": [doc_data["title"] for doc_data in self.documents.values()],
        }
        return stats