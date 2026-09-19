from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    index_dir: str = "data/index"
    eval_dir: str = "data/eval"


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 500
    overlap: int = 100


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5

    # Retrieval strategy.
    mode: str = "hybrid"

    # Hybrid retrieval parameters.
    hybrid_alpha: float = 0.5
    fusion_method: str = "rrf"
    rrf_k: int = 60

    # Dense retrieval parameters.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity: str = "cosine"
    device: str = "cpu"
    batch_size: int = 32
    normalize_embeddings: bool = True


@dataclass(frozen=True)
class PromptConfig:
    template_name: str = "grounded"
    max_context_chars: int = 4000
    max_chunk_chars: int = 1200


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "mock"
    model_name: str | None = None
    base_url: str | None = None
    temperature: float = 0.2
    max_tokens: int = 512
    thinking_enabled: bool = False
    timeout: int = 60


@dataclass(frozen=True)
class LoggingConfig:
    log_file: str = "logs/app.log"
    level: str = "INFO"


@dataclass(frozen=True)
class AppConfig:
    project_name: str = "PaperPilot-Lite"
    data: DataConfig = DataConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    prompt: PromptConfig = PromptConfig()
    llm: LLMConfig = LLMConfig()
    logging: LoggingConfig = LoggingConfig()


def get_default_config_path() -> Path:
    """Return the repository-level default YAML configuration path."""

    return (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "rag_config.yaml"
    )


def load_config(
    config_path: str | Path | None = None,
) -> AppConfig:
    """Load YAML config and apply environment overrides.

    Priority:
        environment variables > YAML > dataclass defaults
    """

    path = (
        Path(config_path)
        if config_path is not None
        else get_default_config_path()
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Config path is not a file: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise TypeError(
            "Top-level YAML config must be a mapping"
        )

    defaults = AppConfig()

    data_raw = _section(raw, "data")
    chunking_raw = _section(raw, "chunking")
    retrieval_raw = _section(raw, "retrieval")
    prompt_raw = _section(raw, "prompt")
    llm_raw = _section(raw, "llm")
    logging_raw = _section(raw, "logging")

    data = DataConfig(
        raw_dir=_env_str(
            "PAPERPILOT_RAW_DIR",
            data_raw.get(
                "raw_dir",
                defaults.data.raw_dir,
            ),
        ),
        processed_dir=_env_str(
            "PAPERPILOT_PROCESSED_DIR",
            data_raw.get(
                "processed_dir",
                defaults.data.processed_dir,
            ),
        ),
        index_dir=_env_str(
            "PAPERPILOT_INDEX_DIR",
            data_raw.get(
                "index_dir",
                defaults.data.index_dir,
            ),
        ),
        eval_dir=_env_str(
            "PAPERPILOT_EVAL_DIR",
            data_raw.get(
                "eval_dir",
                defaults.data.eval_dir,
            ),
        ),
    )

    chunking = ChunkingConfig(
        chunk_size=_env_int(
            "PAPERPILOT_CHUNK_SIZE",
            chunking_raw.get(
                "chunk_size",
                defaults.chunking.chunk_size,
            ),
        ),
        overlap=_env_int(
            "PAPERPILOT_OVERLAP",
            chunking_raw.get(
                "overlap",
                defaults.chunking.overlap,
            ),
        ),
    )

    retrieval = RetrievalConfig(
        top_k=_env_int(
            "PAPERPILOT_TOP_K",
            retrieval_raw.get(
                "top_k",
                defaults.retrieval.top_k,
            ),
        ),
        mode=_env_str(
            "PAPERPILOT_RETRIEVAL_MODE",
            retrieval_raw.get(
                "mode",
                defaults.retrieval.mode,
            ),
        ).lower(),
        hybrid_alpha=_env_float(
            "PAPERPILOT_HYBRID_ALPHA",
            retrieval_raw.get(
                "hybrid_alpha",
                defaults.retrieval.hybrid_alpha,
            ),
        ),
        fusion_method=_env_str(
            "PAPERPILOT_FUSION_METHOD",
            retrieval_raw.get(
                "fusion_method",
                defaults.retrieval.fusion_method,
            ),
        ).lower(),
        rrf_k=_env_int(
            "PAPERPILOT_RRF_K",
            retrieval_raw.get(
                "rrf_k",
                defaults.retrieval.rrf_k,
            ),
        ),
        embedding_model=_env_str(
            "PAPERPILOT_EMBEDDING_MODEL",
            retrieval_raw.get(
                "embedding_model",
                defaults.retrieval.embedding_model,
            ),
        ),
        similarity=_env_str(
            "PAPERPILOT_SIMILARITY",
            retrieval_raw.get(
                "similarity",
                defaults.retrieval.similarity,
            ),
        ),
        device=_env_str(
            "PAPERPILOT_DEVICE",
            retrieval_raw.get(
                "device",
                defaults.retrieval.device,
            ),
        ),
        batch_size=_env_int(
            "PAPERPILOT_BATCH_SIZE",
            retrieval_raw.get(
                "batch_size",
                defaults.retrieval.batch_size,
            ),
        ),
        normalize_embeddings=_env_bool(
            "PAPERPILOT_NORMALIZE_EMBEDDINGS",
            retrieval_raw.get(
                "normalize_embeddings",
                defaults.retrieval.normalize_embeddings,
            ),
        ),
    )

    prompt = PromptConfig(
        template_name=_env_str(
            "PAPERPILOT_TEMPLATE_NAME",
            prompt_raw.get(
                "template_name",
                defaults.prompt.template_name,
            ),
        ),
        max_context_chars=_env_int(
            "PAPERPILOT_MAX_CONTEXT_CHARS",
            prompt_raw.get(
                "max_context_chars",
                defaults.prompt.max_context_chars,
            ),
        ),
        max_chunk_chars=_env_int(
            "PAPERPILOT_MAX_CHUNK_CHARS",
            prompt_raw.get(
                "max_chunk_chars",
                defaults.prompt.max_chunk_chars,
            ),
        ),
    )

    llm = LLMConfig(
        provider=_env_str(
            "LLM_PROVIDER",
            llm_raw.get(
                "provider",
                defaults.llm.provider,
            ),
        ).lower(),
        model_name=_env_optional_str(
            "LLM_MODEL_NAME",
            llm_raw.get(
                "model_name",
                defaults.llm.model_name,
            ),
        ),
        base_url=_env_optional_str(
            "LLM_BASE_URL",
            llm_raw.get(
                "base_url",
                defaults.llm.base_url,
            ),
        ),
        temperature=_env_float(
            "LLM_TEMPERATURE",
            llm_raw.get(
                "temperature",
                defaults.llm.temperature,
            ),
        ),
        max_tokens=_env_int(
            "LLM_MAX_TOKENS",
            llm_raw.get(
                "max_tokens",
                defaults.llm.max_tokens,
            ),
        ),
        thinking_enabled=_env_bool(
            "LLM_THINKING_ENABLED",
            llm_raw.get(
                "thinking_enabled",
                defaults.llm.thinking_enabled,
            ),
        ),
        timeout=_env_int(
            "LLM_TIMEOUT",
            llm_raw.get(
                "timeout",
                defaults.llm.timeout,
            ),
        ),
    )

    logging_config = LoggingConfig(
        log_file=_env_str(
            "PAPERPILOT_LOG_FILE",
            logging_raw.get(
                "log_file",
                defaults.logging.log_file,
            ),
        ),
        level=_env_str(
            "PAPERPILOT_LOG_LEVEL",
            logging_raw.get(
                "level",
                defaults.logging.level,
            ),
        ).upper(),
    )

    config = AppConfig(
        project_name=_env_str(
            "PAPERPILOT_PROJECT_NAME",
            raw.get(
                "project_name",
                defaults.project_name,
            ),
        ),
        data=data,
        chunking=chunking,
        retrieval=retrieval,
        prompt=prompt,
        llm=llm,
        logging=logging_config,
    )

    _validate_config(config)

    return config


def _section(
    raw: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    value = raw.get(name, {})

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise TypeError(
            f"Config section '{name}' must be a mapping"
        )

    return value


def _env_str(
    name: str,
    default: Any,
) -> str:
    value = os.getenv(name)

    if value is not None:
        value = value.strip()

        if not value:
            raise ValueError(
                f"Environment variable {name} "
                "must not be empty"
            )

        return value

    if not isinstance(default, str):
        raise TypeError(
            f"Config value for {name} "
            "must be a string"
        )

    if not default.strip():
        raise ValueError(
            f"Config value for {name} "
            "must not be empty"
        )

    return default.strip()


def _env_optional_str(
    name: str,
    default: Any,
) -> str | None:
    value = os.getenv(name)

    candidate = (
        value
        if value is not None
        else default
    )

    if candidate is None:
        return None

    if not isinstance(candidate, str):
        raise TypeError(
            f"Config value for {name} "
            "must be a string or null"
        )

    normalized = candidate.strip()

    return normalized or None


def _env_float(
    name: str,
    default: Any,
) -> float:
    value = os.getenv(name)

    candidate = (
        value
        if value is not None
        else default
    )

    if isinstance(candidate, bool):
        raise TypeError(
            f"Config value for {name} "
            "must be a number"
        )

    try:
        return float(candidate)

    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Config value for {name} "
            f"must be a number: {candidate}"
        ) from exc


def _env_int(
    name: str,
    default: Any,
) -> int:
    value = os.getenv(name)

    candidate = (
        value
        if value is not None
        else default
    )

    if isinstance(candidate, bool):
        raise TypeError(
            f"Config value for {name} "
            "must be an integer"
        )

    try:
        return int(candidate)

    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Config value for {name} "
            f"must be an integer: {candidate}"
        ) from exc


def _env_bool(
    name: str,
    default: Any,
) -> bool:
    value = os.getenv(name)

    candidate = (
        value
        if value is not None
        else default
    )

    if isinstance(candidate, bool):
        return candidate

    if isinstance(candidate, str):
        normalized = candidate.strip().lower()

        if normalized in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return True

        if normalized in {
            "0",
            "false",
            "no",
            "off",
        }:
            return False

    raise ValueError(
        f"Config value for {name} "
        f"must be a boolean: {candidate}"
    )


def _validate_config(
    config: AppConfig,
) -> None:
    if config.chunking.chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive"
        )

    if config.chunking.overlap < 0:
        raise ValueError(
            "overlap must be non-negative"
        )

    if (
        config.chunking.overlap
        >= config.chunking.chunk_size
    ):
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    if config.retrieval.top_k <= 0:
        raise ValueError(
            "top_k must be positive"
        )

    if config.retrieval.mode not in {
        "dense",
        "keyword",
        "hybrid",
    }:
        raise ValueError(
            "retrieval.mode must be one of: "
            "dense, keyword, hybrid"
        )

    if not (
        0.0
        <= config.retrieval.hybrid_alpha
        <= 1.0
    ):
        raise ValueError(
            "retrieval.hybrid_alpha must be "
            "between 0 and 1"
        )

    if config.retrieval.fusion_method not in {
        "minmax",
        "rrf",
    }:
        raise ValueError(
            "retrieval.fusion_method must be "
            "one of: minmax, rrf"
        )

    if config.retrieval.rrf_k <= 0:
        raise ValueError(
            "retrieval.rrf_k must be positive"
        )

    if config.retrieval.batch_size <= 0:
        raise ValueError(
            "batch_size must be positive"
        )

    if config.retrieval.similarity not in {
        "cosine",
        "inner_product",
        "l2",
    }:
        raise ValueError(
            "similarity must be one of: "
            "cosine, inner_product, l2"
        )

    if config.prompt.template_name not in {
        "extractive",
        "grounded",
        "explainer",
    }:
        raise ValueError(
            "template_name must be one of: "
            "extractive, grounded, explainer"
        )

    if config.prompt.max_context_chars <= 0:
        raise ValueError(
            "max_context_chars must be positive"
        )

    if config.prompt.max_chunk_chars <= 0:
        raise ValueError(
            "max_chunk_chars must be positive"
        )

    if config.llm.provider not in {
        "mock",
        "openai-compatible",
    }:
        raise ValueError(
            "llm.provider must be one of: "
            "mock, openai-compatible"
        )

    if config.llm.temperature < 0:
        raise ValueError(
            "llm.temperature must be non-negative"
        )

    if config.llm.max_tokens <= 0:
        raise ValueError(
            "llm.max_tokens must be positive"
        )

    if config.llm.timeout <= 0:
        raise ValueError(
            "llm.timeout must be positive"
        )

    valid_log_levels = {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }

    if (
        config.logging.level
        not in valid_log_levels
    ):
        raise ValueError(
            "logging.level must be one of: "
            + ", ".join(
                sorted(valid_log_levels)
            )
        )