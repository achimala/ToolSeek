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

# ───────────────────────────── config ───────────────────────────── #
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
        "python": "fg:#00af5f",  # green-ish for code
        "output": "fg:#ffaf00",  # orange-ish for run-output
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
    """Incrementally colourises <python>...</python> and <output>...</output> blocks."""

    TAGS = {"python", "output"}

    def __init__(self):
        self.buf: list[str] = []  # holds characters crossing chunk boundaries
        self.active: str | None = None  # current open tag

    def feed(self, text: str) -> list[tuple[str, str | None]]:
        """
        Returns a list of (segment, style_name) tuples ready for printing.
        style_name is None for normal text / reasoning.
        """
        self.buf.append(text)
        data = "".join(self.buf)
        out: list[tuple[str, str | None]] = []
        i = 0
        while i < len(data):
            if data.startswith("<", i):
                # try to see if we have a full tag already
                j = data.find(">", i + 1)
                if j == -1:  # tag not complete yet → keep in buffer
                    break
                tag = data[i + 1 : j].strip().lower().lstrip("/")
                closing = data[i + 1] == "/"
                if tag in self.TAGS:
                    # flush text *before* tag
                    if i:
                        out.append((data[:i], self.active))
                    # update state
                    self.active = None if closing else tag
                    # cut consumed piece
                    data = data[j + 1 :]
                    i = 0
                    continue
            i += 1
        # whatever is left without incomplete tag
        if data:
            out.append((data, self.active))
            self.buf = []
        else:
            self.buf = [data]  # keep remainder (incomplete tag) for next feed
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
                        for seg, kind in tagger.feed(reasoning):
                            style_class = (
                                "python"
                                if kind == "python"
                                else "output" if kind == "output" else "reason"
                            )
                            display(FormattedText([(f"class:{style_class}", seg)]))
                        assistant_accum += reasoning
                    if content:
                        if not has_finished_reasoning:
                            has_finished_reasoning = True
                            display(FormattedText([("class:reason", "\n")]))
                        for seg, kind in tagger.feed(content):
                            style_class = (
                                "python"
                                if kind == "python"
                                else "output" if kind == "output" else ""
                            )
                            display(FormattedText([(f"class:{style_class}", seg)]))
                        assistant_accum += content

                display("\n")  # newline after assistant finishes
                hist.append({"role": "assistant", "content": assistant_accum})

    except KeyboardInterrupt:
        pass
    finally:
        display("\nBye!\n")


if __name__ == "__main__":
    main()
