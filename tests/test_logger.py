from src.paperpilot.logger import get_logger, setup_logging


def test_setup_logging_creates_log_file(tmp_path):
    log_path = tmp_path / "logs" / "paperpilot.log"
    logger = setup_logging(level="INFO", log_file=log_path)

    logger.info("hello paperpilot")

    assert log_path.exists()
    assert "hello paperpilot" in log_path.read_text(encoding="utf-8")


def test_get_logger_returns_child_logger():
    logger = get_logger("retriever")

    assert logger.name == "paperpilot.retriever"
