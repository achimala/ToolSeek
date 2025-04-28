# ToolSeek

This project attempts to create an o3-style reasoner that can use tools inside its CoT by taking DeepSeek R1 and abusing response prefixing to extract and run Python code within the CoT.

Built by [@anshuchimala](https://x.com/anshuchimala). Credit to [@willccbb](https://x.com/willccbb) for the idea!

![ToolSeek Demo](Demo.gif)

This project provides:

1. A FastAPI server that exposes an OpenAI-compatible Chat Completions endpoint (`/v1/chat/completions`) which forwards requests to Deepseek via the OpenAI Python library.
2. A simple CLI tool (`cli.py`) to interact with the server and stream responses in your terminal.

## Requirements

- Python 3.7+
- A Deepseek API key.

## Setup

1. Clone the repository:

   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```

2. Create a `.env` file based on `.env.example` and fill in your Deepseek credentials:

   ```bash
   cp .env.example .env
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

Start the FastAPI server with Uvicorn:

```bash
uvicorn server:app --reload --port 8000
```

The server will listen on `http://localhost:8000`. It provides:

- `POST /v1/chat/completions` – OpenAI-compatible endpoint for chat completions.

## Using the CLI Tool

Run the interactive CLI to chat with the model:

```bash
python cli.py
```

You can set these environment variables to override defaults:

- `LLM_API_URL` – URL of the proxy server endpoint (default: `http://localhost:8000/v1/chat/completions`).

Type your messages when prompted. Responses will stream in real time.

## Disclaimer

This project is provided as-is. No warranty. Use at your own risk. LLM-generated code will be run on your computer without guardrails; I'm not responsible for any damages caused by this.
