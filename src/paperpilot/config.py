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
class RerankerConfig:
    """Cross-Encoder Reranker 配置。"""

    # 默认关闭，保证原有 retrieval baseline 不变。
    enabled: bool = False

    # Passage ranking 模型。
    model_name: str = (
        "cross-encoder/"
        "ms-marco-MiniLM-L-6-v2"
    )

    # 第一阶段交给 reranker 的候选数量。
    candidate_k: int = 10

    # Reranker 推理设备。
    device: str = "cpu"

    # Cross-Encoder predict() 的 batch size。
    batch_size: int = 16


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5

    # Retrieval strategy.
    mode: str = "hybrid"

    # Hybrid retrieval parameters.
    hybrid_alpha: float = 0.5
    fusion_method: str = "rrf"
    rrf_k: int = 60

    # Dense / BM25 各自参与 Hybrid fusion
    # 的候选深度。
    #
    # 与最终 top_k 独立。
    hybrid_candidate_k: int = 20

    # Dense retrieval parameters.
    embedding_model: str = (
        "sentence-transformers/"
        "all-MiniLM-L6-v2"
    )

    similarity: str = "cosine"
    device: str = "cpu"
    batch_size: int = 32
    normalize_embeddings: bool = True

    # Second-stage reranking.
    reranker: RerankerConfig = RerankerConfig()


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

    chunking: ChunkingConfig = (
        ChunkingConfig()
    )

    retrieval: RetrievalConfig = (
        RetrievalConfig()
    )

    prompt: PromptConfig = PromptConfig()

    llm: LLMConfig = LLMConfig()

    logging: LoggingConfig = (
        LoggingConfig()
    )


def get_default_config_path() -> Path:
    """Return repository-level default YAML config path."""

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

        environment variables
        >
        YAML
        >
        dataclass defaults
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

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(
        raw,
        dict,
    ):
        raise TypeError(
            "Top-level YAML config must be a mapping"
        )

    defaults = AppConfig()

    # -------------------------------------------------
    # YAML sections
    # -------------------------------------------------
    data_raw = _section(
        raw,
        "data",
    )

    chunking_raw = _section(
        raw,
        "chunking",
    )

    retrieval_raw = _section(
        raw,
        "retrieval",
    )

    # retrieval:
    #   reranker:
    #     enabled: false
    #     ...
    reranker_raw = _section(
        retrieval_raw,
        "reranker",
    )

    prompt_raw = _section(
        raw,
        "prompt",
    )

    llm_raw = _section(
        raw,
        "llm",
    )

    logging_raw = _section(
        raw,
        "logging",
    )

    # -------------------------------------------------
    # Data
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Chunking
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Reranker
    # -------------------------------------------------
    reranker = RerankerConfig(
        enabled=_env_bool(
            "PAPERPILOT_RERANKER_ENABLED",
            reranker_raw.get(
                "enabled",
                defaults.retrieval.reranker.enabled,
            ),
        ),

        model_name=_env_str(
            "PAPERPILOT_RERANKER_MODEL_NAME",
            reranker_raw.get(
                "model_name",
                defaults.retrieval.reranker.model_name,
            ),
        ),

        candidate_k=_env_int(
            "PAPERPILOT_RERANKER_CANDIDATE_K",
            reranker_raw.get(
                "candidate_k",
                defaults.retrieval.reranker.candidate_k,
            ),
        ),

        device=_env_str(
            "PAPERPILOT_RERANKER_DEVICE",
            reranker_raw.get(
                "device",
                defaults.retrieval.reranker.device,
            ),
        ),

        batch_size=_env_int(
            "PAPERPILOT_RERANKER_BATCH_SIZE",
            reranker_raw.get(
                "batch_size",
                defaults.retrieval.reranker.batch_size,
            ),
        ),
    )

    # -------------------------------------------------
    # Retrieval
    # -------------------------------------------------
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

        hybrid_candidate_k=_env_int(
            "PAPERPILOT_HYBRID_CANDIDATE_K",
            retrieval_raw.get(
                "hybrid_candidate_k",
                defaults.retrieval.hybrid_candidate_k,
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
        ).lower(),

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

        reranker=reranker,
    )

    # -------------------------------------------------
    # Prompt
    # -------------------------------------------------
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

    # -------------------------------------------------
    # LLM
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Logging
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Final AppConfig
    # -------------------------------------------------
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

    _validate_config(
        config
    )

    return config


def _section(
    raw: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    """读取一个 YAML mapping section。"""

    value = raw.get(
        name,
        {},
    )

    if value is None:
        return {}

    if not isinstance(
        value,
        dict,
    ):
        raise TypeError(
            f"Config section '{name}' "
            "must be a mapping"
        )

    return value


def _env_str(
    name: str,
    default: Any,
) -> str:
    value = os.getenv(
        name
    )

    if value is not None:
        value = value.strip()

        if not value:
            raise ValueError(
                f"Environment variable "
                f"{name} must not be empty"
            )

        return value

    if not isinstance(
        default,
        str,
    ):
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
    value = os.getenv(
        name
    )

    candidate = (
        value
        if value is not None
        else default
    )

    if candidate is None:
        return None

    if not isinstance(
        candidate,
        str,
    ):
        raise TypeError(
            f"Config value for {name} "
            "must be a string or null"
        )

    normalized = (
        candidate.strip()
    )

    return normalized or None


def _env_float(
    name: str,
    default: Any,
) -> float:
    value = os.getenv(
        name
    )

    candidate = (
        value
        if value is not None
        else default
    )

    if isinstance(
        candidate,
        bool,
    ):
        raise TypeError(
            f"Config value for {name} "
            "must be a number"
        )

    try:
        return float(
            candidate
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"Config value for {name} "
            f"must be a number: "
            f"{candidate}"
        ) from exc


def _env_int(
    name: str,
    default: Any,
) -> int:
    value = os.getenv(
        name
    )

    candidate = (
        value
        if value is not None
        else default
    )

    if isinstance(
        candidate,
        bool,
    ):
        raise TypeError(
            f"Config value for {name} "
            "must be an integer"
        )

    try:
        return int(
            candidate
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"Config value for {name} "
            f"must be an integer: "
            f"{candidate}"
        ) from exc


def _env_bool(
    name: str,
    default: Any,
) -> bool:
    value = os.getenv(
        name
    )

    candidate = (
        value
        if value is not None
        else default
    )

    if isinstance(
        candidate,
        bool,
    ):
        return candidate

    if isinstance(
        candidate,
        str,
    ):
        normalized = (
            candidate
            .strip()
            .lower()
        )

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
        f"must be a boolean: "
        f"{candidate}"
    )


def _validate_config(
    config: AppConfig,
) -> None:
    """验证完整配置是否合法。"""

    # -------------------------------------------------
    # Chunking
    # -------------------------------------------------
    if (
        config.chunking.chunk_size
        <= 0
    ):
        raise ValueError(
            "chunk_size must be positive"
        )

    if (
        config.chunking.overlap
        < 0
    ):
        raise ValueError(
            "overlap must be non-negative"
        )

    if (
        config.chunking.overlap
        >= config.chunking.chunk_size
    ):
        raise ValueError(
            "overlap must be smaller "
            "than chunk_size"
        )

    # -------------------------------------------------
    # Retrieval
    # -------------------------------------------------
    if (
        config.retrieval.top_k
        <= 0
    ):
        raise ValueError(
            "top_k must be positive"
        )

    if (
        config.retrieval.mode
        not in {
            "dense",
            "keyword",
            "hybrid",
        }
    ):
        raise ValueError(
            "retrieval.mode must be "
            "one of: dense, keyword, hybrid"
        )

    if not (
        0.0
        <= config.retrieval.hybrid_alpha
        <= 1.0
    ):
        raise ValueError(
            "retrieval.hybrid_alpha "
            "must be between 0 and 1"
        )

    if (
        config.retrieval.fusion_method
        not in {
            "minmax",
            "rrf",
        }
    ):
        raise ValueError(
            "retrieval.fusion_method "
            "must be one of: minmax, rrf"
        )

    if (
        config.retrieval.rrf_k
        <= 0
    ):
        raise ValueError(
            "retrieval.rrf_k "
            "must be positive"
        )

    if (
        config.retrieval.hybrid_candidate_k
        <= 0
    ):
        raise ValueError(
            "retrieval.hybrid_candidate_k "
            "must be positive"
        )

    if (
        config.retrieval.batch_size
        <= 0
    ):
        raise ValueError(
            "retrieval.batch_size "
            "must be positive"
        )

    if (
        config.retrieval.similarity
        not in {
            "cosine",
            "inner_product",
            "l2",
        }
    ):
        raise ValueError(
            "retrieval.similarity must be "
            "one of: cosine, "
            "inner_product, l2"
        )

    # -------------------------------------------------
    # Reranker
    # -------------------------------------------------
    reranker = (
        config.retrieval.reranker
    )

    if (
        reranker.candidate_k
        <= 0
    ):
        raise ValueError(
            "retrieval.reranker."
            "candidate_k must be positive"
        )

    if (
        reranker.batch_size
        <= 0
    ):
        raise ValueError(
            "retrieval.reranker."
            "batch_size must be positive"
        )

    # candidate_k >= top_k 只有真正启用
    # reranker 时才是硬约束。
    #
    # 这样用户即使临时设置：
    #
    # top_k = 30
    # reranker.enabled = false
    #
    # 也不会因为一个没有使用的 reranker
    # 配置而导致整个应用启动失败。
    if (
        reranker.enabled
        and reranker.candidate_k
        < config.retrieval.top_k
    ):
        raise ValueError(
            "retrieval.reranker."
            "candidate_k must be greater "
            "than or equal to retrieval.top_k "
            "when reranker is enabled"
        )

    # -------------------------------------------------
    # Prompt
    # -------------------------------------------------
    if (
        config.prompt.template_name
        not in {
            "extractive",
            "grounded",
            "explainer",
        }
    ):
        raise ValueError(
            "template_name must be one of: "
            "extractive, grounded, explainer"
        )

    if (
        config.prompt.max_context_chars
        <= 0
    ):
        raise ValueError(
            "max_context_chars "
            "must be positive"
        )

    if (
        config.prompt.max_chunk_chars
        <= 0
    ):
        raise ValueError(
            "max_chunk_chars "
            "must be positive"
        )

    # -------------------------------------------------
    # LLM
    # -------------------------------------------------
    if (
        config.llm.provider
        not in {
            "mock",
            "openai-compatible",
        }
    ):
        raise ValueError(
            "llm.provider must be "
            "one of: mock, openai-compatible"
        )

    if (
        config.llm.temperature
        < 0
    ):
        raise ValueError(
            "llm.temperature "
            "must be non-negative"
        )

    if (
        config.llm.max_tokens
        <= 0
    ):
        raise ValueError(
            "llm.max_tokens "
            "must be positive"
        )

    if (
        config.llm.timeout
        <= 0
    ):
        raise ValueError(
            "llm.timeout "
            "must be positive"
        )

    # -------------------------------------------------
    # Logging
    # -------------------------------------------------
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
                sorted(
                    valid_log_levels
                )
            )
        )