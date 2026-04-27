import json
import re
import time
from pathlib import Path
from typing import AsyncIterator

from .llm import LLMClient, AnthropicClient, OpenAIClient
from .tools import (
    TOOL_DEFINITIONS, TOOL_FUNCTIONS,
    ToolResult, ToolError, AskUserSignal,
)
from .skills import SkillRegistry


SYSTEM_PROMPT_TEMPLATE = """You are an AI coding agent with full access to a Linux environment.

ABILITIES:
You can read, write, and edit files, execute bash commands, search files with glob and grep, and list directories. Use these tools to explore the codebase and accomplish the user's goals. Prefer exploring before making changes.

RULES:
1. THINK step by step before using any tool. Use the workspace to understand the codebase.
2. For BASH commands:
   - The CWD is {workspace_path}
   - Commands run with a 30-second timeout
   - Do NOT run interactive commands (vim, nano, top, etc.)
   - NEVER run commands that modify system state outside the workspace
3. For FILE operations:
   - Always read a file before editing it
   - Use edit_file for surgical changes, write_file for new files
   - All paths are relative to the workspace root
4. When you need information from the user (clarification, missing details, choices), you MUST call the ask_user tool instead of writing the question as text output. The ask_user tool pauses execution and waits for the user's response. Only use text output for final answers and summaries, never for asking questions back to the user.
5. When you complete a task, summarize what you did
6. ALWAYS respond in Chinese (中文). The user's primary language is Chinese. All explanations, summaries, and responses must be in Chinese, regardless of the language of the code or files being discussed.
7. Your thinking and reasoning process must also be in Chinese (中文). Think step by step in Chinese.

WORKSPACE:
{workspace_path}"""


class AgentSettings:
    def __init__(
        self,
        provider: str = "anthropic",
        api_key: str = "",
        model: str = "",
        base_url: str = "",
        workspace_path: str = "",
        skill_dirs: list[str] | None = None,
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.workspace_path = workspace_path
        self.skill_dirs = skill_dirs or []


class Agent:
    def __init__(self, settings: AgentSettings):
        self.settings = settings
        self.workspace = Path(settings.workspace_path).resolve() if settings.workspace_path else Path.cwd()
        self.messages: list = []
        self.llm: LLMClient | None = None
        self.skills = SkillRegistry(
            [Path(p) for p in settings.skill_dirs] if settings.skill_dirs else []
        )
        self._pending_tool_results: list[dict] = []
        self._pending_thinking: list[dict] = []

    def initialize(self):
        self.skills.discover()
        self.llm = self._create_llm_client()

    def _create_llm_client(self) -> LLMClient:
        s = self.settings
        if s.provider == "openai":
            return OpenAIClient(api_key=s.api_key, model=s.model or "gpt-4o", base_url=s.base_url or None)
        return AnthropicClient(api_key=s.api_key, model=s.model or "claude-sonnet-4-20250514", base_url=s.base_url or None)

    def _build_system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(workspace_path=str(self.workspace))

    def _get_tool_definitions(self):
        tools = list(TOOL_DEFINITIONS)
        tools.extend(self.skills.get_tool_definitions())
        return tools

    # ------------------------------------------------------------------ #
    # Step-by-step agent operations (called by main.py agent_loop)
    # ------------------------------------------------------------------ #

    async def stream_llm_call(self) -> AsyncIterator[dict]:
        """
        Call LLM with current message history.
        Yields events: text_delta, thinking_delta, tool_use_start
        After iteration stops, the parsed result is available via .last_response property.
        """
        self._pending_text = []
        self._pending_tool_uses = []
        self._pending_thinking = []

        async for event in self.llm.stream_messages(
            system=self._build_system_prompt(),
            messages=self.messages,
            tools=self._get_tool_definitions(),
        ):
            etype = event["type"]
            if etype == "thinking_delta":
                yield event
            elif etype == "thinking_complete":
                self._pending_thinking.append({"type": "thinking", "thinking": event["thinking"]})
            elif etype == "text_delta":
                yield event
            elif etype == "text_complete":
                self._pending_text.append({"type": "text", "text": event["text"]})
                yield {"type": "text_end"}
            elif etype == "tool_use_start":
                self._pending_tool_uses.append(event)
                yield {
                    "type": "tool_call_start",
                    "tool_call_id": event["id"],
                    "tool_name": event["name"],
                    "args": event["input"],
                }
            elif etype == "error":
                yield event

    def commit_llm_response(self):
        """Add the LLM's response (thinking + text + tool_uses) to conversation history."""
        content_blocks = list(self._pending_thinking)
        content_blocks.extend(self._pending_text)
        for tu in self._pending_tool_uses:
            content_blocks.append({
                "type": "tool_use",
                "id": tu["id"],
                "name": tu["name"],
                "input": tu["input"],
            })
        self.messages.append({"role": "assistant", "content": content_blocks})

    @property
    def pending_tool_uses(self) -> list[dict]:
        return self._pending_tool_uses

    @property
    def has_tool_uses(self) -> bool:
        return len(self._pending_tool_uses) > 0

    async def execute_tool(self, tool_call_id: str, name: str, args: dict) -> AsyncIterator[dict]:
        """
        Execute a single tool. Yields result events and returns the tool_result dict.
        For ask_user tools, yields the ask_user event — the caller must handle
        injecting the response via inject_ask_user_response() and re-calling LLM.
        """
        start = time.time()

        # Ensure args is a dict
        if not isinstance(args, dict):
            yield {"type": "tool_call_error", "tool_call_id": tool_call_id, "error": f"Invalid args type: {type(args).__name__}"}
            self._inject_tool_result(tool_call_id, f"Error: args must be a dict, got {type(args).__name__}", is_error=True)
            return

        # ask_user
        if name == "ask_user":
            yield {
                "type": "ask_user",
                "tool_call_id": tool_call_id,
                "question": args.get("question", ""),
            }
            return

        # Built-in tools
        if name in TOOL_FUNCTIONS:
            try:
                result = await TOOL_FUNCTIONS[name](workspace=self.workspace, **args)
            except ToolError as e:
                yield {"type": "tool_call_error", "tool_call_id": tool_call_id, "error": str(e)}
                self._inject_tool_result(tool_call_id, f"Error: {e}", is_error=True)
                return
            except Exception as e:
                yield {"type": "tool_call_error", "tool_call_id": tool_call_id, "error": str(e)}
                self._inject_tool_result(tool_call_id, f"Error: {e}", is_error=True)
                return

            elapsed = round(time.time() - start, 1)
            if result.success:
                yield {"type": "tool_call_end", "tool_call_id": tool_call_id, "result": result.output, "elapsed": elapsed}
            else:
                err_msg = result.error or result.output or "unknown"
                yield {"type": "tool_call_error", "tool_call_id": tool_call_id, "error": err_msg, "elapsed": elapsed}

            self._inject_tool_result(
                tool_call_id,
                result.output if result.success else f"Error: {result.error or result.output}",
                is_error=not result.success,
            )
            return

        # Skill tools
        skill_result = await self.skills.execute_skill(name, _workspace=self.workspace, **args)
        yield {"type": "tool_call_end", "tool_call_id": tool_call_id, "result": skill_result}
        self._inject_tool_result(tool_call_id, skill_result)

    def _inject_tool_result(self, tool_call_id: str, content: str, is_error: bool = False):
        self._pending_tool_results.append({
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": content,
            "is_error": is_error,
        })

    def flush_tool_results(self):
        """Flush all pending tool results as a single user message."""
        if self._pending_tool_results:
            self.messages.append({
                "role": "user",
                "content": list(self._pending_tool_results),
            })
            self._pending_tool_results = []

    def inject_ask_user_response(self, response: str, tool_call_id: str):
        """Inject user's response to an ask_user question."""
        self._inject_tool_result(tool_call_id, response)

    async def generate_suggestions(self) -> list[str]:
        """Generate follow-up suggestion questions based on recent conversation."""
        if not self.llm or len(self.messages) < 2:
            return []

        recent = self.messages[-4:]
        context_parts = []
        for m in recent:
            role = m["role"]
            content = m.get("content", "")
            if isinstance(content, list):
                texts = []
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        texts.append(b.get("text", ""))
                content = " ".join(texts)
            content = str(content)[:600]
            context_parts.append(f"[{role}]: {content}")
        context_str = "\n".join(context_parts)

        system = """You generate follow-up questions. Based on the last exchange, suggest 3 concise follow-up questions the user might ask next. Match the conversation language. Return ONLY a valid JSON array of strings, no markdown, no code fences. Example: ["Question 1?", "Question 2?", "Question 3?"]"""

        msgs = [{"role": "user", "content": f"Suggest 3 follow-up questions for this conversation:\n{context_str}"}]

        text_buf = []
        try:
            async for evt in self.llm.stream_messages(system=system, messages=msgs, tools=[]):
                if evt["type"] == "text_delta":
                    text_buf.append(evt["delta"])
                elif evt["type"] == "error":
                    return []
            text = "".join(text_buf).strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                text = text.rsplit("```", 1)[0]
            text = text.strip()
            # Try to find a JSON array in the response
            m = re.search(r'\[.*?\]', text, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                if isinstance(parsed, list) and len(parsed) > 0:
                    return [str(s)[:120] for s in parsed][:5]
        except Exception:
            pass
        return []
