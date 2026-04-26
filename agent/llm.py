import json
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

import httpx


class LLMClient(ABC):
    """Abstract LLM client that yields a unified stream of events."""

    @abstractmethod
    async def stream_messages(
        self,
        system: str,
        messages: list,
        tools: list,
    ) -> AsyncIterator[dict]:
        """
        Yields event dicts with types:
          - {"type": "thinking_delta", "delta": str}
          - {"type": "text_delta", "delta": str}
          - {"type": "text_complete", "text": str}
          - {"type": "tool_use_start", "id": str, "name": str, "input": dict}
          - {"type": "tool_use_complete"}
          - {"type": "error", "message": str}
        """
        ...

    @abstractmethod
    def model_name(self) -> str:
        ...


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", base_url: str | None = None):
        import anthropic
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)
        self._model = model

    def model_name(self) -> str:
        return self._model

    async def stream_messages(
        self,
        system: str,
        messages: list,
        tools: list,
    ) -> AsyncIterator[dict]:
        try:
            async with self._client.messages.stream(
                model=self._model,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=messages,
                tools=tools if tools else None,  # avoid sending empty list
                max_tokens=8192,
            ) as stream:
                current_tool_block = None
                current_text = ""
                has_thinking = False

                async for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            current_tool_block = {
                                "id": event.content_block.id,
                                "name": event.content_block.name,
                                "input": "",
                            }
                        elif event.content_block.type == "text":
                            current_text = ""
                        elif event.content_block.type == "thinking":
                            has_thinking = True

                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            current_text += event.delta.text
                            yield {"type": "text_delta", "delta": event.delta.text}
                        elif event.delta.type == "thinking_delta":
                            yield {"type": "thinking_delta", "delta": event.delta.thinking}
                        elif event.delta.type == "input_json_delta" and current_tool_block:
                            current_tool_block["input"] += event.delta.partial_json

                    elif event.type == "content_block_stop":
                        if current_tool_block:
                            try:
                                input_data = json.loads(current_tool_block["input"]) if current_tool_block["input"] else {}
                            except json.JSONDecodeError:
                                input_data = {}
                            yield {
                                "type": "tool_use_start",
                                "id": current_tool_block["id"],
                                "name": current_tool_block["name"],
                                "input": input_data,
                            }
                            current_tool_block = None
                        elif current_text:
                            yield {"type": "text_complete", "text": current_text}
                            current_text = ""

                # If we had no tool_use and no text blocks, still mark text complete
                if current_text:
                    yield {"type": "text_complete", "text": current_text}

        except Exception as e:
            yield {"type": "error", "message": str(e)}


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str | None = None):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url or "https://api.openai.com/v1"
        self._client = httpx.AsyncClient(timeout=120)

    def model_name(self) -> str:
        return self._model

    async def stream_messages(
        self,
        system: str,
        messages: list,
        tools: list,
    ) -> AsyncIterator[dict]:
        # Convert Anthropic message format to OpenAI format
        oai_messages = [{"role": "system", "content": system}]
        for msg in messages:
            role = msg["role"]
            raw_content = msg.get("content", "")

            if role == "assistant":
                # Content can be a string or a list of blocks
                if isinstance(raw_content, str):
                    oai_messages.append({"role": "assistant", "content": raw_content or ""})
                    continue
                # Separate text content from tool_use blocks
                text_parts = []
                tool_calls = []
                for block in raw_content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })
                    elif block.get("type") == "thinking":
                        pass  # skip Anthropic-specific blocks
                msg_body = {"role": "assistant", "content": "".join(text_parts) or ""}
                if tool_calls:
                    msg_body["tool_calls"] = tool_calls
                oai_messages.append(msg_body)

            elif role == "user":
                # Content can be a string or a list of blocks (with tool_result)
                if isinstance(raw_content, str):
                    oai_messages.append({"role": "user", "content": raw_content})
                    continue
                # Split tool_result blocks into separate tool-role messages
                text_parts = []
                for block in raw_content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        # OpenAI uses separate "tool" role messages
                        oai_messages.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": block.get("content", ""),
                        })
                    elif block.get("type") == "thinking":
                        pass
                if text_parts:
                    oai_messages.append({"role": "user", "content": "".join(text_parts)})
                # If only tool_results, no user message needed

        # Convert tools to OpenAI format
        oai_tools = []
        for t in tools:
            oai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            })

        body = {
            "model": self._model,
            "messages": oai_messages,
            "stream": True,
            "max_tokens": 8192,
        }
        if oai_tools:
            body["tools"] = oai_tools
            body["tool_choice"] = "auto"

        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json=body,
            ) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    yield {"type": "error", "message": f"OpenAI API error {resp.status_code}: {error_text.decode(errors='replace')[:500]}"}
                    return

                current_tool = None
                current_text = ""

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    # Skip the initial OpenAI role marker chunk (only has "role":"assistant", no content/tool_calls)
                    if delta.get("role") and not delta.get("content") and not delta.get("reasoning_content") and not delta.get("tool_calls"):
                        continue

                    # Reasoning content → thinking_delta, also accumulate in current_text
                    rc = delta.get("reasoning_content")
                    if rc:
                        current_text += rc
                        yield {"type": "thinking_delta", "delta": rc}

                    # Text content → text_delta
                    t = delta.get("content")
                    if t:
                        current_text += t
                        yield {"type": "text_delta", "delta": t}

                    # Tool calls
                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            if tc.get("id"):
                                # Start new tool call
                                if current_tool:
                                    try:
                                        input_data = json.loads(current_tool["arguments"]) if current_tool["arguments"] else {}
                                    except json.JSONDecodeError:
                                        input_data = {}
                                    yield {
                                        "type": "tool_use_start",
                                        "id": current_tool["id"],
                                        "name": current_tool["name"],
                                        "input": input_data,
                                    }
                                current_tool = {
                                    "id": tc["id"],
                                    "name": tc["function"]["name"],
                                    "arguments": tc["function"].get("arguments", ""),
                                }
                            elif current_tool:
                                current_tool["arguments"] += tc["function"].get("arguments", "")

                # Flush remaining
                if current_text:
                    yield {"type": "text_complete", "text": current_text}
                    current_text = ""

                if current_tool:
                    try:
                        input_data = json.loads(current_tool["arguments"]) if current_tool["arguments"] else {}
                    except json.JSONDecodeError:
                        input_data = {}
                    yield {
                        "type": "tool_use_start",
                        "id": current_tool["id"],
                        "name": current_tool["name"],
                        "input": input_data,
                    }

        except Exception as e:
            yield {"type": "error", "message": str(e)}
