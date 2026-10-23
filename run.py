"""Терминал: python run.py или python run.py --ask 'Какая погода в Москве?'"""

import argparse
import asyncio

from agent import Agent
from config import ROOT, client, instance_lock, settings, setup_logging


async def main():
    parser = argparse.ArgumentParser(description="VPf07: AI-агент с инструментами")
    parser.add_argument("--ask", help="Один вопрос и выход")
    args = parser.parse_args()
    config = settings()
    async with client(config) as api:
        agent = Agent(api, ROOT / "data", config["model"])
        if args.ask:
            print(await agent.answer("cli", args.ask))
            return
        print("VPf07 AI-агент. /exit — выход, /reset — очистить историю, /files — список файлов.")
        while True:
            question = await asyncio.to_thread(input, "Вы: ")
            if question.strip() == "/exit":
                break
            if question.strip() == "/reset":
                await agent.reset("cli")
                print("История очищена; файлы сохранены.")
                continue
            if question.strip() == "/files":
                print("\n".join(agent.tools("cli").files()) or "Файлов пока нет.")
                continue
            try:
                print("Агент:", await agent.answer("cli", question))
            except Exception as exc:
                print(str(exc) if isinstance(exc, ValueError) else
                      f"Не удалось получить ответ ({type(exc).__name__}). Проверьте API и сеть.")


if __name__ == "__main__":
    setup_logging()
    try:
        with instance_lock():
            asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception as exc:
        print(str(exc) if isinstance(exc, ValueError) else f"Ошибка запуска: {type(exc).__name__}")
        raise SystemExit(1)
