import os
import json
import re
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import AsyncOpenAI
import io
import traceback
from contextlib import redirect_stdout

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


def run_python(src: str, env: dict[str, object]) -> str:
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            # Try eval first so `print(<expr>)` isn't required
            try:
                result = eval(src, env)
                if result is not None:
                    print(repr(result))
            except SyntaxError:
                exec(src, env)
    except Exception:
        traceback.print_exc(file=buf)
    return buf.getvalue().rstrip() or "(no output)"


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

    if not body.get("stream", False):
        raise HTTPException(
            status_code=400, detail="only streaming is currently supported"
        )

    user_message = {
        "role": "user",
        "content": f"""
This is a system message, not written by the user.
You have a special Python tool which lets you run Python code by outputting <python> tags inside your think section.
The code will be extracted and executed, and you will see the output immediately.
You should use this whenever possible, as it's very fast and precise.
You don't need to think or plan out the code before using the tool. Just go straight to <python> tags, then see the results and iterate.
Don't over-optimize for efficiency, the user doesn't mind waiting. Just try to get to the code as quickly as you can.
There are no timeouts or penalties in the system, so use the tool liberally and frequently.

IMPORTANT: You must output <python> tags INSIDE your think section, BEFORE escaping to the user answer.
The system can only execute code BEFORE you complete the thinking section -- this is now part of your reasoning, not part of the answer.

Here is the user query, which you should solve using code execution:

{messages[-1]["content"]}
        """.strip(),
    }

    prefix = """
<think>
Okay, so apparently I can use Python inside this section. Let me try it:

<python>
print("Hello, world!")
</python>
<output>
Hello, world!
</output>

Wow, it works! So I can run code in here now, which is very powerful. I should use this as much as I can. Okay, so let's look at the user's query and see how I can help them using code execution.

Since the instructions said to not worry about efficiency too much and use the tool frequently and liberally, I'll do that and see what happens. Let's see if math works:
<python>
print(21394872138 + 4723928746)
</python>
<output>
26118800884
</output>

Perfect, that is much more efficient than me doing it myself by hand. This will save a lot of time, so I should use it as much as possible.

Okay, let's start working on the user's query with this. Let me write a first attempt in <python> tags, then I'll see the results and iterate.

<python>\n""".lstrip()

    # Forward parameters directly
    params = body.copy()
    params["model"] = "deepseek-reasoner"
    params["stream"] = True

    py_env: dict[str, object] = {}
    is_thinking = True
    needs_restart = False
    start_with_code = True

    async def event_stream():
        nonlocal is_thinking
        nonlocal needs_restart
        nonlocal prefix
        nonlocal start_with_code

        # Tool loop - we will re-run each time the model produces a tool call.
        # We run the code, update the prefix with the output, then restart with the new prefix.
        while True:
            buffer = ""
            already_sent = ""
            maybe_send = ""
            injected_messages = messages[:-1] + [
                user_message,
                {
                    "role": "assistant",
                    "prefix": True,
                    "content": prefix,
                },
            ]
            params["messages"] = injected_messages

            print(f"Making request with params: {params}")
            try:
                async for chunk in await openai.chat.completions.create(**params):
                    if start_with_code:
                        buffer = "<python>\n"
                        already_sent = "<python>\n"
                        start_with_code = False
                        yield f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': '<python>\n', 'content': ''}}]})}\n\n"
                    data = chunk.to_dict()
                    print(f"Received chunk: {data}")

                    # No longer in CoT -> nothing to do, just forward the data
                    if not is_thinking:
                        yield f"data: {json.dumps(data)}\n\n"
                        continue

                    # Update buffer with new content
                    choices = data.get("choices")
                    if choices:
                        delta = choices[0].get("delta", {})

                        if text := delta.get("content"):
                            buffer += text

                            # Emit the delta to the client, up to and including any </python> tags
                            # Process the buffer to handle Python code blocks
                            if "</python>" in buffer:
                                # Only yield up to the </python> tag, the rest will be processed
                                parts = buffer.split("</python>", 1)
                                text_to_yield = parts[0] + "</python>"
                                # Only send what hasn't been sent yet
                                if text_to_yield.startswith(already_sent):
                                    new_content = text_to_yield[len(already_sent) :]
                                    if new_content:
                                        yield f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': new_content, 'content': ''}}]})}\n\n"
                                        already_sent += new_content
                                        prefix += new_content
                            elif any(
                                buffer.endswith("</think"[:i])
                                for i in range(1, len("</think") + 1)
                            ):
                                # If buffer ends with a partial "</think" tag, we don't want to send any of those tokens, only tokens prior to that
                                # Check if the text contains part of the closing tag
                                # Find the position where the partial closing tag starts
                                for i in range(1, len("</think") + 1):
                                    if buffer.endswith("</think"[:i]):
                                        # Yield everything up to the start of the partial tag
                                        text_to_yield = buffer[:-i]
                                        # Only send what hasn't been sent yet
                                        if text_to_yield.startswith(already_sent):
                                            new_content = text_to_yield[
                                                len(already_sent) :
                                            ]
                                            if new_content:
                                                yield f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': new_content, 'content': ''}}]})}\n\n"
                                                already_sent += new_content
                                                prefix += new_content
                                        break
                                # Skip until the closing tag is complete
                                maybe_send += text
                                continue
                            elif "</think>" in buffer:
                                # Only yield up to the </think> tag
                                parts = buffer.split("</think>", 1)
                                text_to_yield = parts[0]
                                # Only send what hasn't been sent yet
                                if text_to_yield.startswith(already_sent):
                                    new_content = text_to_yield[len(already_sent) :]
                                    if new_content:
                                        yield f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': new_content, 'content': ''}}]})}\n\n"
                                        already_sent += new_content
                                        prefix += new_content
                                # We're done with the thinking section
                                is_thinking = False
                                # For simplicity for now, we just restart the tool loop
                                prefix += text_to_yield + "</think>\n"
                                needs_restart = True
                                break
                            else:
                                if maybe_send:
                                    yield f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': maybe_send, 'content': ''}}]})}\n\n"
                                    already_sent += maybe_send
                                    prefix += maybe_send
                                    maybe_send = ""
                                yield f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': text, 'content': ''}}]})}\n\n"
                                already_sent += text
                                prefix += text
                                continue

                            # Look for complete Python code block
                            # Find the last occurrence of <python>...</python> in the buffer
                            last_start = (
                                buffer.rindex("<python>")
                                if "<python>" in buffer
                                else -1
                            )
                            last_end = (
                                buffer.rindex("</python>")
                                if "</python>" in buffer
                                else -1
                            )
                            if (
                                last_start == -1
                                or last_end == -1
                                or last_start > last_end
                            ):
                                continue

                            # Extract the content between the last <python> and </python> tags
                            code = buffer[last_start + len("<python>") : last_end]

                            output = run_python(code, py_env)
                            formatted_output = f"\n<output>\n{output}\n</output>"
                            prefix += formatted_output

                            # Yield the output to the client
                            yield f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': formatted_output, 'content': ''}}]})}\n\n"
                            already_sent += formatted_output

                            # Restart with the new prefix
                            needs_restart = True
                            break

                if needs_restart:
                    needs_restart = False
                    continue

                # signal end of stream
                yield "data: [DONE]\n\n"
            except Exception as e:
                # In case of error, send error message and close
                err = {"error": {"message": str(e)}}
                yield f"data: {json.dumps(err)}\n\n"
                yield "data: [DONE]\n\n"
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")
