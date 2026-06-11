from src.paperpilot.evaluator import check_hit, summarize_results


def test_check_hit_by_source_file():
    results = [
        {
            "text": "Some text about RAG.",
            "metadata": {"file_name": "rag_intro.txt"},
            "score": 0.9,
        }
    ]

    item = {
        "question": "What is RAG?",
        "expected_source_file": "rag_intro.txt",
        "expected_keywords": ["not existing keyword"],
    }

    hit_info = check_hit(results, item)

    assert hit_info["hit"] is True
    assert hit_info["source_hit"] is True


def test_check_hit_by_keyword():
    results = [
        {
            "text": "Recall@K checks whether expected evidence appears in top K results.",
            "metadata": {"file_name": "sample_rag.pdf"},
            "score": 0.8,
        }
    ]

    item = {
        "question": "What does Recall@K measure?",
        "expected_source_file": "wrong_file.txt",
        "expected_keywords": ["Recall@K"],
    }

    hit_info = check_hit(results, item)

    assert hit_info["hit"] is True
    assert hit_info["keyword_hit"] is True


def test_check_hit_returns_false_when_no_match():
    results = [
        {
            "text": "This text is about attention.",
            "metadata": {"file_name": "attention.md"},
            "score": 0.7,
        }
    ]

    item = {
        "question": "What is RAG?",
        "expected_source_file": "rag_intro.txt",
        "expected_keywords": ["Retrieval-Augmented Generation"],
    }

    hit_info = check_hit(results, item)

    assert hit_info["hit"] is False


def test_summarize_results():
    results = [
        {
            "text": "A long text about RAG.",
            "metadata": {
                "file_name": "rag_intro.txt",
                "page": 1,
                "chunk_id": "rag_intro.txt:1:0",
            },
            "score": 0.9,
        }
    ]

    summaries = summarize_results(results)

    assert len(summaries) == 1
    assert summaries[0]["rank"] == 1
    assert summaries[0]["file_name"] == "rag_intro.txt"
    assert summaries[0]["page"] == 1
    assert summaries[0]["chunk_id"] == "rag_intro.txt:1:0"