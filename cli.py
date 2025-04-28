#!/usr/bin/env python3

from __future__ import annotations
import json
import os
import re
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

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import Terminal256Formatter

_PY_LEXER = PythonLexer(ensurenl=False, stripnl=False)
_FMT = Terminal256Formatter(style="monokai")  # any 256-colour style works

API_URL = os.getenv("LLM_API_URL", "http://localhost:8000/v1/chat/completions")
HISTORY_FILE = Path.home() / ".toolseek_cli_history"
SLASH_CMDS = ("clear", "exit", "quit", "help")

CLI_STYLE = Style.from_dict(
    {
        "reason": "fg:#888888 italic",
        "explanation": "fg:#888888",
        "prompt": "bold cyan",
        "cmd": "ansiyellow",
        "handle": "bold fg:#b434eb",
        "python": "fg:#00af5f",
        "output": "fg:#ffaf00",
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


class TagStreamer:
    """
    Incrementally detects <python>, </python>, <output>, </output> tags in a
    byte-by-byte stream, keeps track of which block we’re in, and returns
    tuples (text, kind) where kind is:
        • "python" - inside a <python> … </python> block
        • "output" – inside an <output> … </output> block
        • None     – tag lines themselves, or normal reasoning text
    """

    _TAG_RX = re.compile(r"<(/?)(python|output)>", re.IGNORECASE)
    _VALID = {"python", "output"}

    def __init__(self):
        self._buf: str = ""
        self.active: str | None = None

    def feed(self, chunk: str) -> list[tuple[str, str | None]]:
        self._buf += chunk
        out: list[tuple[str, str | None]] = []

        # 1) pull out all *complete* tags first
        while True:
            m = self._TAG_RX.search(self._buf)
            if not m:
                break
            start, end = m.span()

            # emit text before the tag under current style
            if start:
                out.append((self._buf[:start], self.active))

            # emit the tag itself as neutral (None)
            tag_txt = self._buf[start:end]
            out.append((tag_txt, None))

            # update active block
            closing, name = m.groups()
            name = name.lower()
            if name in self._VALID:
                self.active = None if closing else name

            self._buf = self._buf[end:]

        # 2) flush any trailing text that we *know* isn't the start of a real tag
        if self._buf:
            idx = self._buf.find("<")
            if idx >= 0:
                # if next char after '<' can't start a tag, dump everything
                if idx + 1 >= len(self._buf) or not re.match(
                    r"[\/A-Za-z]", self._buf[idx + 1]
                ):
                    out.append((self._buf, self.active))
                    self._buf = ""
                else:
                    # possible real tag: emit text up to that '<', keep the rest buffered
                    safe = self._buf[:idx]
                    if safe:
                        out.append((safe, self.active))
                    self._buf = self._buf[idx:]
            else:
                # no '<' at all → safe to flush all
                out.append((self._buf, self.active))
                self._buf = ""

        return out


# ─────────────────────────── main loop ─────────────────────────── #
def main() -> None:
    display(
        FormattedText([("class:prompt", "ToolSeek CLI"), ("", "  (/help for help)\n")])
    )

    display(
        FormattedText(
            [
                (
                    "class:explanation",
                    "This is an experimental DeepSeek wrapper that can execute Python code in its CoT.\n",
                ),
                (
                    "class:explanation",
                    "Made by ",
                ),
                (
                    "class:handle",
                    "@anshuchimala",
                ),
                (
                    "class:explanation",
                    " (credit to ",
                ),
                (
                    "class:handle",
                    "@willccbb",
                ),
                (
                    "class:explanation",
                    " for the idea)\n\n",
                ),
            ]
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

                spin_evt = threading.Event()
                spin_thread = threading.Thread(
                    target=spinner, args=(spin_evt,), daemon=True
                )
                spin_thread.start()

                try:
                    resp = requests.post(API_URL, json=payload, stream=True, timeout=60)
                    resp.raise_for_status()
                except Exception as e:
                    spin_evt.set()
                    spin_thread.join()
                    display(f"\nRequest error: {e}\n")
                    continue

                assistant_accum = ""
                reasoning_done = False
                started = False
                tagger = TagStreamer()

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

                    if not started:
                        spin_evt.set()
                        spin_thread.join()
                        display(FormattedText([("class:prompt", "AI"), ("", ": ")]))
                        started = True

                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    reasoning = delta.get("reasoning_content")
                    content = delta.get("content")

                    if reasoning:
                        for seg, kind in tagger.feed(reasoning):
                            if kind == "python":
                                display(ANSI(highlight(seg, _PY_LEXER, _FMT)))
                            else:
                                style = "output" if kind == "output" else "reason"
                                display(FormattedText([(f"class:{style}", seg)]))
                        assistant_accum += reasoning

                    if content:
                        if not reasoning_done:
                            reasoning_done = True
                            display(FormattedText([("class:reason", "\n")]))
                        for seg, kind in tagger.feed(content):
                            display(FormattedText([(f"class:output", seg)]))
                        assistant_accum += content

                display("\n")
                hist.append({"role": "assistant", "content": assistant_accum})

    except KeyboardInterrupt:
        pass
    finally:
        display("\nBye!\n")


if __name__ == "__main__":
    main()
