import os
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY must be set in environment")

openai = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

app = FastAPI(title="DeepSeek LLM Proxy", version="0.1.0")

# Enable CORS for all origins (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible Chat Completions endpoint that forwards requests to DeepSeek.
    Supports both streaming and non-streaming.
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON") from e

    # Forward parameters directly
    params = body.copy()
    params["model"] = "deepseek-reasoner"

    stream = params.get("stream", False)

    if stream:

        async def event_stream():
            try:
                async for chunk in await openai.chat.completions.create(**params):
                    data = chunk.to_dict()
                    yield f"data: {json.dumps(data)}\n\n"
                # signal end of stream
                yield "data: [DONE]\n\n"
            except Exception as e:
                # In case of error, send error message and close
                err = {"error": {"message": str(e)}}
                yield f"data: {json.dumps(err)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        try:
            response = await openai.chat.completions.create(**params)
            return JSONResponse(content=response.to_dict())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
