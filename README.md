# Deepseek LLM Proxy

This project provides:
1. A FastAPI server that exposes an OpenAI-compatible Chat Completions endpoint (`/v1/chat/completions`) which forwards requests to Deepseek via the OpenAI Python library.
2. A simple CLI tool (`cli.py`) to interact with the server and stream responses in your terminal.

## Requirements

- Python 3.7+
- A Deepseek API key and base URL.

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

Example non-streaming request:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "gpt-3.5-turbo",
           "messages": [{"role": "user", "content": "Hello!"}],
           "stream": false
         }'
```

Example streaming request:

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "gpt-3.5-turbo",
           "messages": [{"role": "user", "content": "Hello!"}],
           "stream": true
         }'
```

## Using the CLI Tool

Run the interactive CLI to chat with the model:

```bash
python cli.py
```

You can set these environment variables to override defaults:

- `LLM_API_URL` – URL of the proxy server endpoint (default: `http://localhost:8000/v1/chat/completions`).
- `LLM_MODEL` – Model name to use (default: `gpt-3.5-turbo`).

Type your messages when prompted. Responses will stream in real time.

## License

This project is provided as-is. No warranty. Use at your own risk.