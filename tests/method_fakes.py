"""Invented replies and a fake embedding model for the method tests, so nothing is sent or paid for."""

from __future__ import annotations

from collections import deque
from pathlib import Path

from langchain_core.embeddings import Embeddings

from tiered_taxonomy.llm import Reply, parse_json_object
from tiered_taxonomy.retrieval import (
    EmbeddingStore,
)
from tiered_taxonomy.taxonomy import Taxonomy, from_paths


def reply(content: str) -> Reply:
    return Reply(content, parse_json_object(content), 1, 1, 0.0, False)


class Scripted:
    """A client that gives prepared replies in order and remembers what it was asked."""

    def __init__(self, *contents: str) -> None:
        self.replies = deque(contents)
        self.asked: list[str] = []

    def ask(self, _system: str, user: str) -> Reply:
        self.asked.append(user)
        return reply(self.replies.popleft() if self.replies else "{}")


class WordEmbeddings(Embeddings):
    """Embeds a text by counting a few known words, and counts how many texts it was asked to embed."""

    WORDS = ("economy", "politics", "health", "water")

    def __init__(self) -> None:
        self.embedded = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded += len(texts)
        return [[float(text.lower().count(word)) + 0.01 for word in self.WORDS] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def tree() -> Taxonomy:
    return from_paths(["Context->Economy", "Context->Politics", "Health->Water", "Health->Clinics"])


def store(tmp_path: Path) -> tuple[EmbeddingStore, WordEmbeddings]:
    model = WordEmbeddings()
    return EmbeddingStore(model, "fake", tmp_path / "vectors.sqlite"), model
