from pathlib import Path

import pytest

from src.paperpilot.config import load_config


def write_config(path: Path) -> None:
    path.write_text(
        """
project_name: PaperPilot-Test

data:
  raw_dir: data/raw
  processed_dir: data/processed
  index_dir: data/index
  eval_dir: data/eval

chunking:
  chunk_size: 500
  overlap: 100

retrieval:
  top_k: 5
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  similarity: cosine
  device: cpu
  batch_size: 32
  normalize_embeddings: true

prompt:
  template_name: grounded
  max_context_chars: 4000
  max_chunk_chars: 1200

logging:
  log_file: logs/app.log
  level: INFO
""".strip(),
        encoding="utf-8",
    )


def test_load_config_reads_yaml(tmp_path):
    config_path = tmp_path / "rag_config.yaml"
    write_config(config_path)

    config = load_config(config_path)

    assert config.project_name == "PaperPilot-Test"
    assert config.chunking.chunk_size == 500
    assert config.retrieval.top_k == 5
    assert config.prompt.template_name == "grounded"


def test_environment_overrides_yaml(tmp_path, monkeypatch):
    config_path = tmp_path / "rag_config.yaml"
    write_config(config_path)
    monkeypatch.setenv("PAPERPILOT_TOP_K", "9")
    monkeypatch.setenv("PAPERPILOT_CHUNK_SIZE", "700")
    monkeypatch.setenv("PAPERPILOT_LOG_LEVEL", "debug")

    config = load_config(config_path)

    assert config.retrieval.top_k == 9
    assert config.chunking.chunk_size == 700
    assert config.logging.level == "DEBUG"


def test_invalid_overlap_is_rejected(tmp_path):
    config_path = tmp_path / "rag_config.yaml"
    write_config(config_path)
    text = config_path.read_text(encoding="utf-8").replace("overlap: 100", "overlap: 500")
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="overlap must be smaller"):
        load_config(config_path)
