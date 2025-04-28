import os
import json
import re
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY must be set in environment")

openai = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/beta")

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

    # Validate and inject special system message
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="`messages` field must be a list")
    for m in messages:
        if m.get("role") == "system":
            raise HTTPException(
                status_code=400, detail="system messages are currently not supported"
            )

    wrapped_final_message = {
        "role": "user",
        "content": f"""
This is a system message, not written by the user.
You have a special Python tool which lets you run Python code by outputting <python> tags inside your think section.
The code will be extracted and executed, and you will see the output immediately.
You should use this whenever possible, as it's very fast and precise.
You don't need to think or plan before using the tool. Use it first, then see the results and iterate.

IMPORTANT: You must output <python> tags INSIDE your think section, BEFORE escaping to the user answer.
The system can only execute code BEFORE you complete the thinking section -- this is now part of your reasoning, not part of the answer.

---

{messages[-1]["content"]}
        """.strip(),
    }
    injected_messages = messages[:-1] + [
        wrapped_final_message,
        {
            "role": "assistant",
            "prefix": True,
            "content": """
<think>
Okay, so apparently I can use Python inside this section. Let me try it:
<python>
print("Hello, world!")
</python>
<output>
Hello, world!
</output>
Wow, it works! Okay, so let's look at the user's query and see how I can help them using this tool.
""".strip(),
        },
    ]

    # Forward parameters directly
    params = body.copy()
    params["messages"] = injected_messages
    params["model"] = "deepseek-reasoner"

    stream = params.get("stream", False)

    if stream:

        async def event_stream():
            buffer = ""
            try:
                async for chunk in await openai.chat.completions.create(**params):
                    data = chunk.to_dict()
                    # Update buffer with new content
                    for choice in data.get("choices", []):
                        delta = choice.get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            buffer += text

                            # Look for complete Python code blocks
                            while True:
                                match = re.search(
                                    r"<python>(.*?)</python>", buffer, re.DOTALL
                                )
                                if not match:
                                    break

                                code = match.group(1)
                                print(f"Extracted python code: {code}")

                                # Remove the extracted code block from buffer
                                start, end = match.span()
                                buffer = buffer[:start] + buffer[end:]

                    yield f"data: {json.dumps(data)}\n\n"

                # Check for any remaining code in buffer at end of stream
                while True:
                    match = re.search(r"<python>(.*?)</python>", buffer, re.DOTALL)
                    if not match:
                        break
                    code = match.group(1)
                    print(f"Extracted python code at end: {code}")
                    start, end = match.span()
                    buffer = buffer[:start] + buffer[end:]

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
