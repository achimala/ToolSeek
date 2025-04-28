#!/usr/bin/env python3

from __future__ import annotations
import json
import os
import sys
import threading
import time
from itertools import cycle
from pathlib import Path
from typing import List

import requests
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import ANSI, FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

# ───────────────────────────── config ───────────────────────────── #
API_URL = os.getenv("LLM_API_URL", "http://localhost:8000/v1/chat/completions")
HISTORY_FILE = Path.home() / ".deepseek_cli_history"
SLASH_CMDS = ("clear", "exit", "quit", "help")

CLI_STYLE = Style.from_dict(
    {
        "reason": "fg:#888888 italic",
        "prompt": "bold cyan",
        "cmd": "ansiyellow",
    }
)

session = PromptSession(
    history=FileHistory(str(HISTORY_FILE)),
    completer=WordCompleter([f"/{c}" for c in SLASH_CMDS], ignore_case=True),
    style=CLI_STYLE,
)

ChatHist = List[dict]


# ─────────────────────────── utilities ─────────────────────────── #
def display(text: str | FormattedText, end: str = "", flush: bool = True) -> None:
    """Safe wrapper around print_formatted_text inside patch_stdout()."""
    print_formatted_text(text, end=end, style=CLI_STYLE, flush=flush)


def spinner(stop_evt: threading.Event) -> None:
    """
    Ultra-simple CLI spinner.

    Runs in its own *non-async* thread and writes directly to stdout.
    Nothing else to clean up: when `stop_evt.set()` is called the loop
    breaks, blanks the line, flushes, and the thread dies.
    """
    frames = cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
    while not stop_evt.is_set():
        sys.stdout.write(f"\rLoading… {next(frames)}")
        sys.stdout.flush()
        time.sleep(0.08)
    # clear the spinner line
    sys.stdout.write("\r" + " " * 30 + "\r")
    sys.stdout.flush()


def show_help() -> None:
    display(
        FormattedText(
            [
                ("class:cmd", "/clear"),
                ("", " - reset conversation\n"),
                ("class:cmd", "/exit"),
                ("", " - quit\n"),
                ("class:cmd", "/help"),
                ("", " - this help\n"),
            ]
        ),
        end="\n",
    )


def handle_slash(cmd: str, hist: ChatHist) -> bool:
    """Return False if the command should terminate the program."""
    name = cmd[1:].strip().lower()
    if name in ("exit", "quit"):
        return False
    if name == "clear":
        hist.clear()
        # clear the terminal screen
        os.system("clear" if os.name == "posix" else "cls")
        display("Context cleared.\n")
    elif name == "help":
        show_help()
    else:
        display(f"Unknown command: {cmd}\n")
    return True


# ─────────────────────────── main loop ─────────────────────────── #
def main() -> None:
    display(
        FormattedText(
            [("class:prompt", "Deepseek LLM CLI"), ("", "  (/help for help)\n")]
        )
    )

    hist: ChatHist = []

    try:
        with patch_stdout():
            while True:
                try:
                    user_msg = session.prompt([("class:prompt", "You"), ("", ": ")])
                except (EOFError, KeyboardInterrupt):
                    display("\n")
                    break

                if user_msg.startswith("/"):
                    if not handle_slash(user_msg, hist):
                        break
                    continue

                hist.append({"role": "user", "content": user_msg})
                payload = {"messages": hist, "stream": True}

                stop_spin = threading.Event()
                th = threading.Thread(target=spinner, args=(stop_spin,), daemon=True)
                th.start()

                try:
                    resp = requests.post(API_URL, json=payload, stream=True, timeout=60)
                    resp.raise_for_status()
                except Exception as e:
                    stop_spin.set()
                    th.join()
                    display(f"\nRequest error: {e}\n")
                    continue

                assistant_accum = ""
                has_finished_reasoning = False
                has_started_streaming = False
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    # ready to start printing assistant reply
                    if not has_started_streaming:
                        stop_spin.set()
                        th.join()
                        display(FormattedText([("class:prompt", "AI"), ("", ": ")]))
                        has_started_streaming = True

                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    reasoning = delta.get("reasoning_content")
                    content = delta.get("content")

                    if reasoning:
                        display(FormattedText([("class:reason", reasoning)]))
                        assistant_accum += reasoning
                    if content:
                        if not has_finished_reasoning:
                            has_finished_reasoning = True
                            display(FormattedText([("class:reason", "\n")]))
                        display(content)
                        assistant_accum += content

                display("\n")  # newline after assistant finishes
                hist.append({"role": "assistant", "content": assistant_accum})

    except KeyboardInterrupt:
        pass
    finally:
        display("\nBye!\n")


if __name__ == "__main__":
    main()
