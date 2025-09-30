import logging
import sys
from pathlib import Path

def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    use_console: bool = True
) -> None:
    """
    Настраивает корневой логгер для всего приложения.
    
    Args:
        level: уровень логирования (logging.INFO, logging.DEBUG и т.д.)
        log_file: путь к файлу для записи логов (опционально)
        use_console: выводить ли логи в консоль
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    handlers = []

    if use_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    for handler in handlers:
        root_logger.addHandler(handler)