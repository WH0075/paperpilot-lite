from argparse import Namespace

from src.paperpilot import cli


class FakeRetriever:
    def __init__(self, *args, **kwargs):
        pass



def test_handle_eval_uses_args_ks(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        cli.Retriever,
        "from_index",
        classmethod(lambda cls, **kwargs: FakeRetriever()),
    )
    monkeypatch.setattr(cli, "load_qa_set", lambda path: [{"question": "q", "expected_keywords": ["x"]}])

    def fake_evaluate_retrieval(*, retriever, qa_items, ks):
        captured["ks"] = ks
        return {
            "total": 1,
            "ks": list(ks),
            "hit_counts": {k: 0 for k in ks},
            "recall": {k: 0.0 for k in ks},
            "cases": [],
            "failed_cases": [],
        }

    monkeypatch.setattr(cli, "evaluate_retrieval", fake_evaluate_retrieval)
    monkeypatch.setattr(cli, "print_evaluation_report", lambda **kwargs: None)

    args = Namespace(
        ks=[1, 3, 5],
        index_dir="data/index",
        model_name="fake-model",
        device="cpu",
        normalize_embeddings=True,
        batch_size=32,
        qa_path="data/eval/qa_set.jsonl",
        show_failed_cases=False,
        max_failed_cases=10,
    )

    cli.handle_eval(args)

    assert captured["ks"] == [1, 3, 5]
