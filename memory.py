"""История всех успешных диалогов в JSON; последние 20 сообщений — контекст."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path


class Memory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"version": 1, "sessions": {}})
        self._read()  # Повреждённую историю не перезаписываем молча.

    def _read(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("version") != 1 or not isinstance(data.get("sessions"), dict):
            raise ValueError("Неверный формат memory.json. Сохраните копию и проверьте файл.")
        return data

    def _write(self, data):
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)

    def context(self, session):
        messages = self._read()["sessions"].get(str(session), [])[-20:]
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    def append(self, session, question, answer):
        data = self._read()
        history = data["sessions"].setdefault(str(session), [])
        now = datetime.now(timezone.utc).isoformat()
        history.extend([{"role": "user", "content": question, "time": now},
                        {"role": "assistant", "content": answer, "time": now}])
        self._write(data)

    def reset(self, session):
        data = self._read()
        data["sessions"].pop(str(session), None)
        self._write(data)
