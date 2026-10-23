"""Проверки без сети и расходования API: python -m unittest -v."""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import httpx
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.types import Update, Message, User, Chat

from agent import Agent, SEARCH_TOOL, response_text
from bot import create_router
from memory import Memory
from tools import Tools, ToolError, MAX_BYTES


class FilesAndMemory(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tools = Tools(self.root / "files")

    def test_utf8_roundtrip(self):
        result = self.tools.file_write("notes/test.txt", "Привет, мир!")
        self.assertTrue(result["saved"])
        self.assertEqual(self.tools.file_read("notes/test.txt")["content"], "Привет, мир!")

    def test_forbidden_paths(self):
        for name in ("../.env", "/tmp/test.txt", ".env", "test.py", "a/../../test.txt", ""):
            with self.subTest(name=name), self.assertRaises(ToolError):
                self.tools.file_write(name, "test")

    def test_symlink_cannot_escape(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.tools.directory / "link").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ToolError):
            self.tools.file_write("link/test.txt", "no")

    def test_size_limit_in_bytes(self):
        with self.assertRaises(ToolError):
            self.tools.file_write("big.txt", "я" * (MAX_BYTES // 2 + 1))

    def test_missing_file(self):
        with self.assertRaises(ToolError):
            self.tools.file_read("missing.txt")

    def test_persistence_and_user_isolation(self):
        path = self.root / "memory.json"
        memory = Memory(path)
        memory.append("alice", "Меня зовут Виктор", "Запомнил")
        restored = Memory(path)
        self.assertEqual(restored.context("alice")[0]["content"], "Меня зовут Виктор")
        self.assertEqual(restored.context("bob"), [])
        restored.reset("alice")
        self.assertEqual(restored.context("alice"), [])

    def test_context_bounded_but_disk_history_complete(self):
        path = self.root / "memory.json"
        memory = Memory(path)
        for i in range(12):
            memory.append("a", str(i), str(i))
        self.assertEqual(len(memory.context("a")), 20)
        self.assertEqual(len(json.loads(path.read_text())["sessions"]["a"]), 24)

    def test_corrupt_memory_is_not_overwritten(self):
        path = self.root / "memory.json"
        path.write_text("broken")
        with self.assertRaises(ValueError):
            Memory(path)
        self.assertEqual(path.read_text(), "broken")


class AgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.api = NS(responses=NS(create=AsyncMock()))
        self.agent = Agent(self.api, Path(self.temp.name), "gpt-6-luna")

    def final(self, text="Готово"):
        return NS(output=[], output_text=text, status="completed")

    def call(self, name, args):
        return NS(output=[NS(type="function_call", name=name, arguments=json.dumps(args), call_id="one")],
                  output_text="", status="completed")

    async def test_model_chooses_tool_and_receives_result(self):
        self.api.responses.create.side_effect = [self.call("file_write", {"filename": "test.txt", "content": "Привет"}), self.final()]
        trace = []
        await self.agent.answer("alice", "Сохрани приветствие", trace)
        self.assertTrue(trace[0]["result"]["ok"])
        self.assertEqual(self.agent.tools("alice").file_read("test.txt")["content"], "Привет")
        self.assertEqual(self.agent.tools("bob").files(), [])
        self.assertEqual(self.api.responses.create.call_args.kwargs["input"][-1]["type"], "function_call_output")

    async def test_tool_error_is_sent_back_to_model(self):
        self.api.responses.create.side_effect = [self.call("file_read", {"filename": "../secret.txt"}), self.final("Доступ запрещён")]
        trace = []
        self.assertEqual(await self.agent.answer("a", "Прочти файл", trace), "Доступ запрещён")
        self.assertFalse(trace[0]["result"]["ok"])

    async def test_http_failure_is_not_fabricated(self):
        request = httpx.Request("GET", "https://example.org")
        error = httpx.HTTPStatusError("SECRET", request=request, response=httpx.Response(429, request=request))
        self.api.responses.create.side_effect = [self.call("get_crypto_price", {"coin": "bitcoin", "currency": "usd"}), self.final("HTTP 429")]
        trace = []
        with patch.object(Tools, "get_crypto_price", AsyncMock(side_effect=error)):
            await self.agent.answer("a", "Курс BTC?", trace)
        self.assertIn("429", trace[0]["result"]["error"])
        self.assertNotIn("SECRET", trace[0]["result"]["error"])

    async def test_history_loaded_after_restart(self):
        self.api.responses.create.return_value = self.final()
        await self.agent.answer("a", "Меня зовут Виктор")
        restarted = Agent(self.api, Path(self.temp.name), self.agent.model)
        await restarted.answer("a", "Как меня зовут?")
        sent = self.api.responses.create.call_args.kwargs["input"]
        self.assertEqual(sent[0]["content"], "Меня зовут Виктор")

    async def test_incomplete_answer_not_saved(self):
        self.api.responses.create.return_value = NS(output=[], output_text="обрыв", status="incomplete")
        with self.assertRaises(ValueError):
            await self.agent.answer("a", "Вопрос")
        self.assertEqual(self.agent.memory.context("a"), [])

    async def test_invalid_session(self):
        with self.assertRaises(ValueError):
            self.agent.tools("../other")

    async def test_native_search_enabled_and_traced(self):
        search = NS(type="web_search_call", status="completed", action=NS(model_dump=lambda: {"type": "search"}))
        self.api.responses.create.return_value = NS(output=[search], output_text="Документация: https://python.org", status="completed")
        trace = []
        answer = await self.agent.answer("a", "Найди Python", trace)
        self.assertIn("https://python.org", answer)
        self.assertIn(SEARCH_TOOL, self.api.responses.create.call_args.kwargs["tools"])
        self.assertTrue(trace[0]["result"]["ok"])
        self.assertEqual(trace[0]["tool"], "web_search")

    async def test_search_then_local_file_in_same_turn(self):
        first = self.call("file_write", {"filename": "source.txt", "content": "https://python.org"})
        first.output.insert(0, NS(type="web_search_call", status="completed", action=None))
        self.api.responses.create.side_effect = [first, self.final()]
        trace = []
        await self.agent.answer("a", "Найди и запиши ссылку", trace)
        self.assertEqual([x["tool"] for x in trace], ["web_search", "file_write"])
        self.assertEqual(self.agent.tools("a").file_read("source.txt")["content"], "https://python.org")

    async def test_search_budget_disables_further_search(self):
        first = self.call("file_read", {"filename": "missing.txt"})
        first.output[:0] = [NS(type="web_search_call", status="completed", action=None) for _ in range(3)]
        self.api.responses.create.side_effect = [first, self.final()]
        await self.agent.answer("a", "Выполни поиск")
        self.assertNotIn(SEARCH_TOOL, self.api.responses.create.call_args.kwargs["tools"])

    def test_citations_become_visible_urls(self):
        text = "Ответ. [citation]"
        annotation = NS(type="url_citation", start_index=7, end_index=len(text), url="https://python.org")
        response = NS(output=[NS(type="message", content=[NS(type="output_text", text=text, annotations=[annotation])])])
        self.assertEqual(response_text(response), "Ответ.  (https://python.org)")

    def test_unannotated_marker_not_shown_to_user(self):
        response = NS(output=[], output_text="Сохранено: https://python.org citeturn0search0")
        self.assertEqual(response_text(response), "Сохранено: https://python.org")

    def test_repeated_known_citation_restores_link(self):
        marker = "citeturn0search0"
        response = NS(output=[], output_text="Документация " + marker)
        self.assertIn("https://python.org", response_text(response, {marker: "https://python.org"}))

    def test_sources_visible_when_final_has_no_annotations(self):
        response = NS(output=[], output_text="Сохранено. citeturn0search0")
        answer = response_text(response, sources=["https://python.org", "javascript:alert(1)"])
        self.assertIn("https://python.org", answer)
        self.assertNotIn("javascript:", answer)
        self.assertNotIn("cite", answer)

    async def test_weather_geocodes_then_fetches_forecast(self):
        def handle(request):
            if request.url.host == "geocoding-api.open-meteo.com":
                self.assertEqual(request.url.params["countryCode"], "RU")
                return httpx.Response(200, json={"results": [
                    {"name": "Москва", "population": 1000, "latitude": 55.75,
                     "longitude": 37.61, "country": "Россия", "admin1": "Москва"}]})
            self.assertEqual(request.url.host, "api.open-meteo.com")
            self.assertEqual(request.url.params["wind_speed_unit"], "ms")
            return httpx.Response(200, json={"timezone": "Europe/Moscow",
                                  "current": {"weather_code": 3, "temperature_2m": 10},
                                  "current_units": {"temperature_2m": "°C"}})
        http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        with patch("tools.httpx.AsyncClient", return_value=http):
            result = await self.agent.tools("a").get_weather("Москва", "ru")
        self.assertEqual(result["conditions"], "пасмурно")

    async def test_unknown_coin_is_not_zero_price(self):
        http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
        with patch("tools.httpx.AsyncClient", return_value=http):
            with self.assertRaisesRegex(ToolError, "не нашёл"):
                await self.agent.tools("a").get_crypto_price("missing-coin", "usd")


class RecordingSession(BaseSession):
    """Telegram API в тестах записывает вызовы, ничего не отправляет в Telegram."""
    def __init__(self):
        super().__init__()
        self.calls = []

    async def close(self):
        pass

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        return Message(message_id=len(self.calls), date=datetime.now(timezone.utc),
                       chat=Chat(id=101, type="private"), text=getattr(method, "text", None))

    async def stream_content(self, *args, **kwargs):
        if False:
            yield b""


class TelegramTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.agent = Agent(None, Path(self.temp.name), "test")
        self.agent.answer = AsyncMock(return_value="Ответ агента")
        self.session = RecordingSession()
        self.bot = Bot("123456:TEST_TOKEN_FOR_OFFLINE_TESTS_ONLY", session=self.session)
        self.dp = Dispatcher()
        self.dp.include_router(create_router(self.agent))

    async def send(self, text, chat_type="private"):
        message = Message(message_id=1, date=datetime.now(timezone.utc), chat=Chat(id=101, type=chat_type),
                          from_user=User(id=101, is_bot=False, first_name="Test"), text=text)
        await self.dp.feed_update(self.bot, Update(update_id=1, message=message))

    async def test_start_and_natural_language(self):
        await self.send("/start")
        self.assertIn("OpenAI Web Search", self.session.calls[-1].text)
        await self.send("Какая погода в Москве?")
        self.agent.answer.assert_awaited_once_with("101", "Какая погода в Москве?")
        self.assertEqual(self.session.calls[-1].text, "Ответ агента")

    async def test_reset_preserves_files(self):
        self.agent.memory.append("101", "hi", "hello")
        self.agent.tools("101").file_write("keep.txt", "keep")
        await self.send("/reset")
        self.assertEqual(self.agent.memory.context("101"), [])
        self.assertEqual(self.agent.tools("101").files(), ["keep.txt"])

    async def test_files_and_download(self):
        self.agent.tools("101").file_write("note.txt", "sample")
        await self.send("/files")
        self.assertIn("note.txt", self.session.calls[-1].text)
        await self.send("/get note.txt")
        self.assertEqual(self.session.calls[-1].__api_method__, "sendDocument")

    async def test_group_ignored(self):
        await self.send("Привет", "group")
        self.agent.answer.assert_not_awaited()
        self.assertEqual(self.session.calls, [])


if __name__ == "__main__":
    unittest.main()
