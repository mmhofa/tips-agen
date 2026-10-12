"""Send one static hello message to the configured Telegram channel."""

import json
import os
import ssl
import truststore

truststore.inject_into_ssl()

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def load_env_file() -> None:
    """Load simple KEY=VALUE entries from the project .env without dependencies."""
    env_file = Path(__file__).with_name(".env")
    if not env_file.exists():
        raise SystemExit("Missing .env. Copy .env.example to .env and set its values.")

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def main() -> None:
    load_env_file()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    channel = os.environ.get("TELEGRAM_CHANNEL", "").strip()
    if not token or not channel:
        raise SystemExit("TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL must be set in .env")

    # Telegram's chat_id accepts @channel_username, not a t.me URL.
    if channel.startswith("https://t.me/"):
        channel = "@" + channel.removeprefix("https://t.me/").strip("/")
    elif channel.startswith("t.me/"):
        channel = "@" + channel.removeprefix("t.me/").strip("/")

    body = json.dumps({"chat_id": channel, "text": "new hello from windows"}).encode("utf-8")
    request = Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        context = ssl.create_default_context()
        with urlopen(request, timeout=20, context=context) as response:
            result = json.load(response)
    except HTTPError as exc:
        # Do not print the request URL because it contains the bot token.
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Telegram API returned HTTP {exc.code}: {detail}") from None
    except URLError as exc:
        raise SystemExit(f"Could not reach Telegram API: {exc.reason}") from None

    if not result.get("ok"):
        raise SystemExit(f"Telegram API rejected the message: {result}")
    message = result["result"]
    print(f"Telegram accepted message {message['message_id']} in chat {message['chat']['id']}.")


if __name__ == "__main__":
    main()
 