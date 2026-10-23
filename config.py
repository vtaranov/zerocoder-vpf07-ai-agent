"""Настройки, журнал без секретов и блокировка второго процесса."""

import logging
import os
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent


def settings():
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError("Заполните OPENAI_API_KEY в .env")
    return {
        "api_key": os.environ["OPENAI_API_KEY"],
        "base_url": os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
        "model": os.getenv("CHAT_MODEL") or "gpt-6-luna",
    }


def client(config):
    return AsyncOpenAI(api_key=config["api_key"], base_url=config["base_url"],
                       timeout=60, max_retries=1)


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(ROOT / "agent.log", encoding="utf-8")])
    for name in ("httpx", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


@contextmanager
def instance_lock():
    # Один процесс на data/memory.json. ОС освобождает lock даже после аварии.
    import fcntl  # macOS / Linux
    with (ROOT / ".instance.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Этот проект уже запущен. Сначала остановите другой CLI/бот.") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
