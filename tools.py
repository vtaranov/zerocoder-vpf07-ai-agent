"""Пять возможностей ДЗ: поиск, погода, криптовалюты, файлы, память (отдельно)."""

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx

MAX_BYTES = 100_000
EXTENSIONS = {".txt", ".md", ".json", ".csv"}
WEATHER_CODES = {
    0: "ясно", 1: "преимущественно ясно", 2: "переменная облачность", 3: "пасмурно",
    45: "туман", 48: "изморозевый туман", 51: "слабая морось", 53: "морось", 55: "сильная морось",
    56: "слабая переохлаждённая морось", 57: "сильная переохлаждённая морось",
    61: "слабый дождь", 63: "дождь", 65: "сильный дождь",
    66: "слабый переохлаждённый дождь", 67: "сильный переохлаждённый дождь",
    71: "слабый снег", 73: "снег", 75: "сильный снег", 77: "снежные зёрна",
    80: "слабый ливень", 81: "ливень", 82: "сильный ливень",
    85: "слабый снежный ливень", 86: "сильный снежный ливень",
    95: "гроза", 96: "гроза со слабым градом", 99: "гроза с сильным градом",
}


def function(name, description, **properties):
    return {"type": "function", "name": name, "description": description, "strict": True,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(properties), "additionalProperties": False}}


def string(description):
    return {"type": "string", "description": description}


SCHEMAS = [
    function("get_weather", "Текущая погода по названию города через Open-Meteo. Не прогноз на завтра.",
             city=string("Название города в именительном падеже, например Москва или Tokyo"),
             country_code=string("Код страны ISO из двух букв, например RU, JP, либо пустая строка")),
    function("get_crypto_price", "Текущий курс криптовалюты по CoinGecko.",
             coin=string("CoinGecko ID, например bitcoin, ethereum, litecoin; поддерживаются BTC, ETH, LTC"),
             currency=string("Валюта, например usd, eur, rub")),
    function("file_read", "Прочитать текстовый файл пользователя из его папки; можно сначала узнать имена из списка в инструкции.",
             filename=string("Относительное имя .txt, .md, .json или .csv")),
    function("file_write", "Создать или перезаписать текстовый файл в папке пользователя. Успех подтверждается только результатом инструмента.",
             filename=string("Относительное имя .txt, .md, .json или .csv"),
             content=string("Полный текст для сохранения, UTF-8")),
]


class ToolError(ValueError):
    pass


class Tools:
    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, filename):
        requested = Path(filename)
        if not filename or requested.is_absolute() or any(p.startswith(".") for p in requested.parts):
            raise ToolError("Используйте относительное имя без скрытых папок и '..'.")
        path = (self.directory / requested).resolve()
        if not path.is_relative_to(self.directory) or path.suffix.lower() not in EXTENSIONS:
            raise ToolError("Доступны только .txt, .md, .json, .csv в вашей папке.")
        return path

    def files(self):
        return [p.relative_to(self.directory).as_posix() for p in self.directory.rglob("*")
                if p.is_file() and not p.is_symlink() and p.suffix.lower() in EXTENSIONS
                and p.resolve().is_relative_to(self.directory)][:100]

    def file_write(self, filename, content):
        path = self.path(filename)
        payload = content.encode("utf-8")
        if len(payload) > MAX_BYTES:
            raise ToolError("Файл больше 100 000 байт.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return {"filename": path.relative_to(self.directory).as_posix(), "bytes": len(payload), "saved": True}

    def file_read(self, filename):
        path = self.path(filename)
        if not path.is_file():
            raise ToolError("Файл не найден. Доступные файлы: " + ", ".join(self.files()))
        with path.open("rb") as handle:
            payload = handle.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            raise ToolError("Файл больше 100 000 байт.")
        try:
            return {"filename": filename, "content": payload.decode("utf-8")}
        except UnicodeDecodeError:
            raise ToolError("Файл должен быть в UTF-8.") from None

    async def get_weather(self, city, country_code):
        if not city.strip() or len(city) > 100:
            raise ToolError("Укажите название города до 100 символов.")
        params = {"name": city, "count": 5, "language": "ru", "format": "json"}
        if country_code:
            if not re.fullmatch(r"[A-Za-z]{2}", country_code):
                raise ToolError("Код страны должен состоять из двух букв.")
            params["countryCode"] = country_code.upper()
        async with httpx.AsyncClient(timeout=25) as http:
            response = await http.get("https://geocoding-api.open-meteo.com/v1/search", params=params)
            response.raise_for_status()
            places = response.json().get("results", [])
            if not places:
                raise ToolError("Город не найден. Уточните название и страну.")
            # Среди точных совпадений выбираем крупнейший город; остальные указываем в ответе.
            exact = [p for p in places if p["name"].casefold() == city.casefold()]
            place = max(exact or places, key=lambda p: p.get("population", 0))
            response = await http.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": place["latitude"], "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                "timezone": "auto", "wind_speed_unit": "ms"})
            response.raise_for_status()
            weather = response.json()
        return {"source": "https://open-meteo.com/", "city": place["name"],
                "country": place.get("country"), "region": place.get("admin1"),
                "latitude": place["latitude"], "longitude": place["longitude"],
                "timezone": weather["timezone"], "current": weather["current"],
                "conditions": WEATHER_CODES.get(weather["current"]["weather_code"], "неизвестный код погоды"),
                "units": weather["current_units"],
                "other_matches": [{"city": p["name"], "country": p.get("country"),
                                   "region": p.get("admin1")} for p in places if p != place]}

    async def get_crypto_price(self, coin, currency):
        aliases = {"btc": "bitcoin", "eth": "ethereum", "ltc": "litecoin", "sol": "solana",
                   "биткоин": "bitcoin", "эфириум": "ethereum"}
        coin = aliases.get(coin.strip().lower(), coin.strip().lower())
        currency = currency.strip().lower()
        if not re.fullmatch(r"[a-z0-9-]{1,80}", coin) or not re.fullmatch(r"[a-z]{2,10}", currency):
            raise ToolError("Укажите CoinGecko ID монеты и код валюты, например bitcoin / usd.")
        async with httpx.AsyncClient(timeout=25) as http:
            response = await http.get("https://api.coingecko.com/api/v3/simple/price", params={
                "ids": coin, "vs_currencies": currency, "include_last_updated_at": "true"})
            response.raise_for_status()
            item = response.json().get(coin, {})
        if not isinstance(item.get(currency), (float, int)):
            raise ToolError("CoinGecko не нашёл монету или валюту. Уточните CoinGecko ID и код валюты.")
        updated = item.get("last_updated_at")
        return {"coin": coin, "currency": currency, "price": item[currency],
                "updated_at_utc": datetime.fromtimestamp(updated, timezone.utc).isoformat() if updated else None,
                "source": "https://www.coingecko.com/en/coins/" + coin}

    async def execute(self, name, arguments):
        if name not in {t["name"] for t in SCHEMAS}:
            raise ToolError("Неизвестный инструмент.")
        result = getattr(self, name)(**arguments)
        return await result if asyncio.iscoroutine(result) else result
