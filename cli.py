#!/usr/bin/env python3
"""
Simple CLI tool to interact with the Deepseek LLM Proxy server.
Sends chat messages and streams responses in the terminal.
"""
import os
import sys
import json
import requests


def main():
    api_url = os.getenv("LLM_API_URL", "http://localhost:8000/v1/chat/completions")

    history = []

    print(
        "Deepseek LLM CLI. Type your message and press Enter (Ctrl-D or Ctrl-C to exit)."
    )
    try:
        while True:
            try:
                message = input("You: ")
            except EOFError:
                print("\nExiting.")
                break

            history.append({"role": "user", "content": message})
            payload = {"messages": history, "stream": True}

            try:
                response = requests.post(api_url, json=payload, stream=True)
                response.raise_for_status()
            except Exception as e:
                print(f"Request error: {e}", file=sys.stderr)
                continue

            assistant_response = ""
            print("AI: ", end="", flush=True)
            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    data = decoded[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            print(content, end="", flush=True)
                            assistant_response += content
                    except json.JSONDecodeError:
                        # skip non-JSON lines
                        continue
            print()
            history.append({"role": "assistant", "content": assistant_response})

    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
