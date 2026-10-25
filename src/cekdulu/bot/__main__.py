"""Bot Telegram mode long-polling (jalur cadangan tanpa n8n/webhook publik).

Jalankan: ``python -m cekdulu.bot`` (butuh CEKDULU_TELEGRAM_BOT_TOKEN).
Gunakan SALAH SATU: workflow n8n (webhook) ATAU poller ini. Telegram menolak getUpdates
bila webhook aktif, jadi poller menghapus webhook saat start.
"""

from __future__ import annotations

import logging
import time

import httpx

from cekdulu.api.container import build_container
from cekdulu.config import get_settings
from cekdulu.logging_setup import configure_logging

logger = logging.getLogger("cekdulu.bot")


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    if not settings.telegram_bot_token:
        raise SystemExit("CEKDULU_TELEGRAM_BOT_TOKEN belum diisi.")
    token = settings.telegram_bot_token.get_secret_value()
    api = f"https://api.telegram.org/bot{token}"
    container = build_container(settings)
    http = httpx.Client(timeout=40)
    http.post(f"{api}/deleteWebhook")
    offset = 0
    logger.info("bot_polling_started")
    while True:
        try:
            resp = http.get(f"{api}/getUpdates", params={"timeout": 30, "offset": offset})
            updates = resp.json().get("result", [])
        except httpx.HTTPError as exc:
            logger.warning("poll_error", extra={"error": str(exc)})
            time.sleep(3)
            continue
        for update in updates:
            offset = update["update_id"] + 1
            chat = (update.get("message") or update.get("edited_message") or {}).get("chat", {})
            if "id" not in chat:
                continue
            try:
                reply = container.telegram.handle(update)
                http.post(
                    f"{api}/sendMessage",
                    json={
                        "chat_id": chat["id"],
                        "text": reply.reply_text,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                )
                if reply.report_id:
                    rec = container.repo.get_report(reply.report_id)
                    if rec:
                        http.post(
                            f"{api}/sendDocument",
                            data={"chat_id": chat["id"]},
                            files={
                                "document": (
                                    f"laporan-{reply.report_id[:8]}.pdf",
                                    rec.pdf,
                                    "application/pdf",
                                )
                            },
                        )
            except Exception:
                logger.exception("handle_update_failed")


if __name__ == "__main__":
    main()
