"""Реальные платные OpenAI-запросы + публичные API. Запускать вручную.

Telegram: только getMe/getWebhookInfo, без отправки сообщений.
История и файлы проверки изолированы от рабочих данных пользователя.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from aiogram import Bot

from agent import Agent
from config import ROOT, client, settings


async def main():
    config = settings()
    results = {"checked_at": datetime.now(ZoneInfo("Europe/Moscow")).isoformat(),
               "model": config["model"], "checks": []}
    async with client(config) as api, Bot(os.environ["BOT_TOKEN"]) as bot:
        me = await bot.get_me()
        results["telegram"] = {"username": me.username, "get_me": True,
                               "webhook_set": bool((await bot.get_webhook_info()).url)}
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(api, Path(directory), config["model"])
            scenarios = [
                ("remember", "Запомни: меня зовут Виктор, я изучаю Python. Ответь кратко."),
                ("recall_after_restart", "Как меня зовут и что я изучаю?"),
                ("weather", "Какая сейчас погода в Москве, Россия? Получи свежие данные, укажи время и источник."),
                ("crypto", "Сколько сейчас стоит 1 Bitcoin в USD? Проверь инструментом, укажи время и источник."),
                ("crypto_followup", "А Ethereum в той же валюте?"),
                ("write", "Создай notes.txt с текстом: Я изучаю Python и AI-агентов. Сегодня я проверил работу инструментов."),
                ("read_write", "Прочитай notes.txt и сохрани краткое резюме в summary.txt."),
                ("search", "Найди через веб-поиск официальную документацию Python asyncio. Дай две ссылки на документацию и кратко объясни, для чего она нужна."),
                ("search_save", "Выполни новый веб-поиск официальной документации Open-Meteo Geocoding API. Сохрани найденное название и URL в sources.txt, используя file_write."),
            ]
            for name, question in scenarios:
                if name == "recall_after_restart":
                    agent = Agent(api, Path(directory), config["model"])
                trace = []
                metrics = []
                print("Checking:", name, flush=True)
                try:
                    answer = await agent.answer("demo", question, trace, metrics)
                    row = {"name": name, "question": question, "answer": answer, "trace": trace,
                           "metrics": metrics}
                    successful = [t["tool"] for t in trace if t["result"]["ok"]]
                    required = {"weather": ["get_weather"], "crypto": ["get_crypto_price"],
                                "crypto_followup": ["get_crypto_price"], "write": ["file_write"],
                                "read_write": ["file_read", "file_write"], "search": ["web_search"],
                                "search_save": ["web_search", "file_write"]}.get(name, [])
                    row["passed"] = all(t in successful for t in required)
                    if name == "recall_after_restart":
                        row["passed"] = "Виктор" in answer and "Python" in answer
                    if name == "crypto_followup":
                        row["passed"] &= any(t["arguments"].get("currency") == "usd" for t in trace)
                    if name == "search":
                        row["passed"] &= "https://" in answer
                    if name == "search_save":
                        row["passed"] &= "https://" in agent.tools("demo").file_read("sources.txt")["content"]
                    print("Passed:", row["passed"], flush=True)
                except Exception as exc:
                    row = {"name": name, "passed": False, "error_type": type(exc).__name__}
                    print("Failed:", type(exc).__name__, flush=True)
                results["checks"].append(row)
            results["files"] = {name: agent.tools("demo").file_read(name)["content"]
                                for name in agent.tools("demo").files()}
            results["memory_message_count"] = len(json.loads((Path(directory) / "memory.json").read_text())["sessions"].get("demo", []))
    out = ROOT / "examples" / "live_check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if not all(c["passed"] for c in results["checks"]):
        raise SystemExit(1)
    print("Saved examples/live_check.json", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print("Live check failed:", type(exc).__name__)
        raise SystemExit(1)
