"""End-to-end test of `runq -m server` with the official OpenAI Python client.

    ./runq model.bin [-z tokenizer.bin] -m server -P 18080 &
    pip install openai
    python3 tests/server_test.py http://127.0.0.1:18080/v1
"""
import json
import sys
import time

from openai import OpenAI, BadRequestError

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080/v1"
client = OpenAI(base_url=base, api_key=sys.argv[2] if len(sys.argv) > 2 else "none")
model = client.models.list().data[0].id
print("model:", model)
ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  ({info})" if info else ""))


# 1. plain completion
r = client.chat.completions.create(model=model, temperature=0, max_tokens=30,
                                   messages=[{"role": "user", "content": "Name the capital of Japan in one word."}])
check("plain", "tokyo" in r.choices[0].message.content.lower(), r.choices[0].message.content)

# 2. streaming == non-streaming with greedy sampling, usage chunk
msgs = [{"role": "system", "content": "You are concise."},
        {"role": "user", "content": "List three primary colors, comma separated."}]
full = client.chat.completions.create(model=model, temperature=0, max_tokens=40, messages=msgs).choices[0].message.content
parts, usage, finish = [], None, None
for ch in client.chat.completions.create(model=model, temperature=0, max_tokens=40, messages=msgs, stream=True,
                                         stream_options={"include_usage": True}):
    if ch.usage: usage = ch.usage
    if ch.choices:
        if ch.choices[0].delta.content: parts.append(ch.choices[0].delta.content)
        if ch.choices[0].finish_reason: finish = ch.choices[0].finish_reason
check("stream matches", "".join(parts) == full, repr("".join(parts)))
check("stream usage", usage is not None and usage.completion_tokens > 0 and finish == "stop", f"{usage} {finish}")

# 3. multi-turn: the shared prefix is served from the KV cache
msgs2 = msgs + [{"role": "assistant", "content": full}, {"role": "user", "content": "Now one more color."}]
r = client.chat.completions.create(model=model, temperature=0, max_tokens=20, messages=msgs2)
cached = r.usage.prompt_tokens_details.cached_tokens
check("prefix cache", cached >= r.usage.prompt_tokens // 2, f"{cached}/{r.usage.prompt_tokens} cached")

# 4. max_tokens -> finish_reason length
r = client.chat.completions.create(model=model, max_tokens=5,
                                   messages=[{"role": "user", "content": "Write a long essay about the sea."}])
check("length", r.choices[0].finish_reason == "length" and r.usage.completion_tokens == 5, r.usage.completion_tokens)

# 5. stop string
r = client.chat.completions.create(model=model, temperature=0, max_tokens=60, stop=["5"],
                                   messages=[{"role": "user", "content": "Count from 1 to 10, separated by spaces."}])
c = r.choices[0].message.content
check("stop", "5" not in c and "4" in c and r.choices[0].finish_reason == "stop", repr(c))

# 6. non-ASCII streaming (multi-byte characters split across tokens must arrive whole)
txt = "".join(ch.choices[0].delta.content or "" for ch in client.chat.completions.create(
    model=model, temperature=0, max_tokens=40, stream=True,
    messages=[{"role": "user", "content": "Переведи на китайский: «Доброе утро, друг». Только перевод."}]) if ch.choices)
check("utf-8 stream", "�" not in txt and len(txt) > 0, txt)

# 7. tool call round trip
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather in a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
q = [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}]
r = client.chat.completions.create(model=model, temperature=0, max_tokens=200, messages=q, tools=tools)
calls = r.choices[0].message.tool_calls
check("tool call", calls and calls[0].function.name == "get_weather" and r.choices[0].finish_reason == "tool_calls",
      calls and calls[0].function.arguments)
if calls:
    args = json.loads(calls[0].function.arguments)
    check("tool args", "paris" in str(args.get("city", "")).lower(), args)
    q += [r.choices[0].message.model_dump(exclude_none=True),
          {"role": "tool", "tool_call_id": calls[0].id, "content": json.dumps({"temp_c": 17, "sky": "light rain"})}]
    r = client.chat.completions.create(model=model, temperature=0, max_tokens=80, messages=q, tools=tools)
    a = r.choices[0].message.content or ""
    check("tool answer", "17" in a, a)
    # streamed tool call
    got = None
    for ch in client.chat.completions.create(model=model, temperature=0, max_tokens=200, messages=q[:1], tools=tools, stream=True):
        if ch.choices and ch.choices[0].delta.tool_calls: got = ch.choices[0].delta.tool_calls[0]
    check("stream tool call", got and got.function.name == "get_weather" and got.id, got)

# 8. errors
try:
    client.chat.completions.create(model=model, messages=[{"role": "user", "content": "x " * 200000}])
    check("too long -> 400", False)
except BadRequestError as e:
    check("too long -> 400", "context" in str(e), str(e)[:80])
try:
    client.chat.completions.create(model=model, messages=[])
    check("empty -> 400", False)
except BadRequestError:
    check("empty -> 400", True)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
