"""Check that the strict QA labels match the pinned paper index."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from src.paperpilot.evaluator import load_qa_set


ROOT = Path(__file__).resolve().parents[1]
PAPERS_PATH = ROOT / "data" / "benchmark" / "papers.json"
QA_PATH = ROOT / "data" / "eval" / "papers_qa_set.jsonl"
CHUNKS_PATH = ROOT / "data" / "benchmark" / "index" / "chunks.json"

QUERY_TYPES = {
    "exact_keyword",
    "semantic_paraphrase",
    "terminology",
    "abbreviation",
    "ambiguous",
    "difficult",
    "lexical_mismatch",
    "rare_identifier",
}


def main() -> None:
    manifest = json.loads(PAPERS_PATH.read_text(encoding="utf-8"))
    papers = manifest["papers"]
    filenames = {paper["filename"] for paper in papers}
    chunks_digest = hashlib.sha256(CHUNKS_PATH.read_bytes()).hexdigest()
    if chunks_digest != manifest["chunks_sha256"]:
        raise ValueError("Indexed chunks differ from the benchmark manifest")
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    chunks_by_id = {
        chunk["metadata"]["chunk_id"]: chunk
        for chunk in chunks
    }
    qa_items = load_qa_set(QA_PATH)

    if not 60 <= len(qa_items) <= 100:
        raise ValueError("Benchmark must contain 60 to 100 questions")

    ids = [item.get("id") for item in qa_items]
    questions = [item["question"] for item in qa_items]
    if len(ids) != len(set(ids)) or len(questions) != len(set(questions)):
        raise ValueError("Benchmark IDs and questions must be unique")

    for item in qa_items:
        source = item.get("expected_source_file")
        if source not in filenames:
            raise ValueError(f"Unknown source for {item['id']}: {source}")
        if item.get("query_type") not in QUERY_TYPES:
            raise ValueError(f"Invalid query type for {item['id']}")
        if not isinstance(item.get("answer"), str) or not item["answer"].strip():
            raise ValueError(f"Missing reference answer for {item['id']}")

        for chunk_id in item["expected_chunk_ids"]:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                raise ValueError(f"Unknown chunk for {item['id']}: {chunk_id}")
            if chunk["metadata"]["file_name"] != source:
                raise ValueError(f"Wrong source for {item['id']}: {chunk_id}")

    print(f"Questions: {len(qa_items)}")
    print(f"Papers: {len(filenames)}")
    print(f"Indexed chunks: {len(chunks)}")
    print(f"Query types: {dict(sorted(Counter(item['query_type'] for item in qa_items).items()))}")


if __name__ == "__main__":
    main()
