"""Telegram-интерфейс того же агента. Запуск: .venv/bin/python bot.py."""

import asyncio
import io
import logging
import os

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandStart
from aiogram.types import FSInputFile, Message

from agent import Agent
from config import ROOT, client, instance_lock, settings, setup_logging
from tools import EXTENSIONS, MAX_BYTES, ToolError

HELP = """Я AI-агент урока VPf07. Умею:
• искать информацию через OpenAI Web Search;
• узнавать текущую погоду по городу;
• получать курс криптовалют;
• читать и записывать текстовые файлы;
• помнить историю после перезапуска.

Напишите задачу обычным языком. Например:
«Какая сейчас погода в Москве?»
«Сколько стоит Bitcoin в USD? А Ethereum?»
«Создай plan.txt с планом изучения Python»
«Прочитай plan.txt и кратко перескажи»

Можно загрузить файл .txt, .md, .json, .csv в UTF-8 (до 100 000 байт).
/files — список файлов
/get имя.txt — скачать файл
/reset — очистить свою историю (файлы останутся)
/help — эта справка"""


def create_router(agent, allowed_ids=None):
    router = Router()
    router.message.filter(F.chat.type == "private")
    if allowed_ids:
        router.message.filter(F.from_user.id.in_(allowed_ids))

    @router.message(CommandStart())
    @router.message(Command("help"))
    async def start(message: Message):
        await message.answer(HELP)

    @router.message(Command("reset"))
    async def reset(message: Message):
        await agent.reset(str(message.from_user.id))
        await message.answer("История очищена. Ваши файлы сохранены.")

    @router.message(Command("files"))
    async def files(message: Message):
        names = agent.tools(str(message.from_user.id)).files()
        text = "\n".join(names) or "Файлов пока нет."
        for start in range(0, len(text), 4000):
            await message.answer(text[start:start + 4000])

    @router.message(Command("get"))
    async def get_file(message: Message):
        parts = message.text.split(maxsplit=1)
        if len(parts) != 2:
            await message.answer("Пример: /get plan.txt")
            return
        try:
            tools = agent.tools(str(message.from_user.id))
            tools.file_read(parts[1])  # Проверка пути, размера и UTF-8 перед отправкой.
            await message.answer_document(FSInputFile(tools.path(parts[1])))
        except ToolError as exc:
            await message.answer(str(exc))

    @router.message(F.document)
    async def document(message: Message, bot: Bot):
        doc = message.document
        if not doc.file_name or not any(doc.file_name.lower().endswith(s) for s in EXTENSIONS):
            await message.answer("Пришлите .txt, .md, .json или .csv в UTF-8.")
            return
        if doc.file_size is None or doc.file_size > MAX_BYTES:
            await message.answer("Файл должен быть не больше 100 000 байт.")
            return
        async with agent.locks[str(message.from_user.id)]:
            try:
                buffer = io.BytesIO()
                await bot.download(doc, destination=buffer)
                payload = buffer.getvalue()
                if len(payload) > MAX_BYTES:
                    raise ToolError("Файл больше 100 000 байт.")
                result = agent.tools(str(message.from_user.id)).file_write(doc.file_name, payload.decode("utf-8"))
                await message.answer(f"Файл сохранён: {result['filename']}. Теперь можно попросить прочитать его.")
            except UnicodeDecodeError:
                await message.answer("Сохраните текст в UTF-8 и загрузите снова.")
            except ToolError as exc:
                await message.answer(str(exc))
            except Exception as exc:
                logging.warning("upload_error=%s", type(exc).__name__)
                await message.answer("Не удалось загрузить файл. Проверьте сеть и повторите.")

    @router.message(F.text)
    async def question(message: Message):
        if message.text.startswith("/"):
            await message.answer(HELP)
            return
        try:
            answer = await agent.answer(str(message.from_user.id), message.text)
        except Exception as exc:
            logging.warning("answer_error=%s", type(exc).__name__)
            await message.answer(str(exc) if isinstance(exc, ValueError) else
                                 "Не удалось получить ответ. Проверьте ключ, баланс API и сеть.")
            return
        for start in range(0, len(answer), 4000):
            await message.answer(answer[start:start + 4000])

    return router


async def main():
    config = settings()
    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        raise ValueError("Заполните BOT_TOKEN в .env")
    allowed = {int(x.strip()) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip()}
    async with client(config) as api, Bot(token=token, default=DefaultBotProperties(link_preview_is_disabled=True)) as bot:
        if (await bot.get_webhook_info()).url:
            raise ValueError("У бота настроен webhook. Отключите его перед polling.")
        me = await bot.get_me()
        agent = Agent(api, ROOT / "data", config["model"])
        dispatcher = Dispatcher()
        dispatcher.include_router(create_router(agent, allowed))
        logging.info("VPf07 started @%s", me.username)
        await dispatcher.start_polling(bot, allowed_updates=["message"])


if __name__ == "__main__":
    setup_logging()
    try:
        with instance_lock():
            asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as exc:
        print(str(exc) if isinstance(exc, ValueError) else f"Ошибка запуска: {type(exc).__name__}")
        raise SystemExit(1)
