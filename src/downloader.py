from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import requests
import streamlit as st

API_URL: str = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
CHUNK: int = 8 * 1024 * 1024  # 8 МБ


def _poll_href(public_url: str, tries: int = 20, pause: float = 1.0) -> str:
    """Запрашиваем прямую ссылку на скачивание (href).
    API может отвечать 202 — в этом случае ждём и повторяем."""
    params: dict[str, str] = {"public_key": public_url}

    for _ in range(tries):
        resp = requests.get(API_URL, params=params, timeout=10)
        if resp.status_code == 200:  # ссылка готова
            return resp.json()["href"]
        if resp.status_code == 202:  # «готовим архив»
            time.sleep(pause)
            continue
        resp.raise_for_status()

    raise RuntimeError("Не удалось получить download-href от Я.Диска")


def download_yadisk(
    public_url: str,
    save_path: str | Path,
) -> Path:
    """Скачивает файл по публичной ссылке Я.Диска."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    href = _poll_href(public_url)

    with requests.get(href, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        progress = st.progress(0, text="⬇️ Скачиваем видео с Я.Диска…")
        written = 0

        with open(save_path, "wb") as f:
            for chunk in r.iter_content(CHUNK):
                if chunk:
                    f.write(chunk)
                    written += len(chunk)
                    if total:
                        progress.progress(min(written / total, 0.98))

        progress.progress(1.0, text="✅ Файл загружен")

    return save_path