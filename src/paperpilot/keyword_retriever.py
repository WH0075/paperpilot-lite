from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

_TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Convert text into normalized keyword tokens."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return [
        token.lower()
        for token in _TOKEN_PATTERN.findall(text)
    ]


class KeywordRetriever:
    """BM25-based keyword retriever over PaperPilot chunks."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty")

        if k1 <= 0:
            raise ValueError("k1 must be positive")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self.tokenized_chunks = [
            tokenize(chunk.get("text", ""))
            for chunk in chunks
        ]

        self.term_frequencies = [
            Counter(tokens)
            for tokens in self.tokenized_chunks
        ]

        self.document_lengths = [
            len(tokens)
            for tokens in self.tokenized_chunks
        ]

        self.num_documents = len(chunks)

        self.average_document_length = (
            sum(self.document_lengths)
            / self.num_documents
        )

        self.document_frequencies: Counter[str] = Counter()

        for tokens in self.tokenized_chunks:
            unique_tokens = set(tokens)

            for token in unique_tokens:
                self.document_frequencies[token] += 1

    def _idf(self, token: str) -> float:
        document_frequency = self.document_frequencies.get(
            token,
            0,
        )

        return math.log(
            1
            + (
                self.num_documents
                - document_frequency
                + 0.5
            )
            / (
                document_frequency
                + 0.5
            )
        )

    def _score_document(
        self,
        query_tokens: list[str],
        document_index: int,
    ) -> float:

        term_frequency = self.term_frequencies[
            document_index
        ]

        document_length = self.document_lengths[
            document_index
        ]

        score = 0.0

        for token in query_tokens:
            frequency = term_frequency.get(token, 0)

            if frequency == 0:
                continue

            idf = self._idf(token)

            numerator = frequency * (self.k1 + 1)

            denominator = (
                frequency
                + self.k1
                * (
                    1
                    - self.b
                    + self.b
                    * document_length
                    / self.average_document_length
                )
            )

            score += idf * numerator / denominator

        return score

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = query.strip()

        if not query:
            raise ValueError("query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_tokens = tokenize(query)

        scored_results: list[
            tuple[int, float]
        ] = []

        for document_index in range(
            self.num_documents
        ):
            score = self._score_document(
                query_tokens,
                document_index,
            )

            scored_results.append(
                (document_index, score)
            )

        scored_results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        results: list[
            dict[str, Any]
        ] = []

        for document_index, score in scored_results[:top_k]:
            chunk = self.chunks[document_index]

            results.append(
                {
                    "text": chunk.get("text", ""),
                    "metadata": dict(
                        chunk.get("metadata", {})
                    ),
                    "score": float(score),
                    "index": document_index,
                }
            )

        return results