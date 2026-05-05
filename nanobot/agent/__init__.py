"""LocalLLMProvider with full debug logging for DashScope"""

import json
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any
import httpx

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """A tool call from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from the LLM."""
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LocalLLMProvider:
    """
    LLM provider using any OpenAI-compatible API.
    
    Works with: Ollama, vLLM, llama.cpp, LM Studio, LocalAI, DashScope, etc.
    """

    def __init__(
        self,
        api_base: str,
        api_key: str = "no-key",
        model: str = "default",
        timeout: float = 600,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        
        # 细粒度超时配置
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
            connect=15.0,
            write=30.0,    # ← 改大，tool result 可能很长
            read=300.0,    # ← thinking 模型需要更长
            pool=10.0,
        )
    )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        extra_body: dict | None = None
    ) -> LLMResponse:
        """Send a chat completion request."""
        url = f"{self.api_base}/chat/completions"

        body: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "enable_thinking": False,
        }

        if extra_body:
            body.update(extra_body)

        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        headers_str = " ".join([f'-H "{k}: {v}"' for k, v in headers.items()])
        body_str = json.dumps(body, ensure_ascii=False)

        try:
            resp = await self._client.post(url, json=body, headers=headers)
            
            resp.raise_for_status()
            data = resp.json()
            return self._parse(data)
        
        except httpx.ConnectError as e:
            print(f"❌ Connection error: {e}")
            logger.error(f"Connection error: {e}")
            return LLMResponse(content=None)
        
        except httpx.TimeoutException as e:
            print(f"❌ Timeout error: {e}")
            logger.error(f"Timeout: {e}")
            return LLMResponse(content=None)
        
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP error {e.response.status_code}: {e.response.text[:300]}")
            logger.error(f"HTTP {e.response.status_code}: {e.response.text[:300]}")
            return LLMResponse(content=None)
        
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            logger.error(f"JSON decode error: {e}")
            return LLMResponse(content=None)
        
        except Exception as e:
            print(f"❌ Unexpected error: {type(e).__name__}: {e}")
            logger.error(f"Unexpected error: {type(e).__name__}: {e}")
            return LLMResponse(content=None)

    def _parse(self, data: dict) -> LLMResponse:
        """Parse OpenAI-format response."""
        try:
            choice = data["choices"][0]
            message = choice["message"]

            tool_calls = []
            if message.get("tool_calls"):
                for tc in message["tool_calls"]:
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"raw": args}
                    tool_calls.append(ToolCall(
                        id=tc.get("id", f"call_{len(tool_calls)}"),
                        name=tc["function"]["name"],
                        arguments=args,
                    ))

            usage = {}
            if data.get("usage"):
                usage = {
                    "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                    "completion_tokens": data["usage"].get("completion_tokens", 0),
                }

            return LLMResponse(
                content=message.get("content"),
                tool_calls=tool_calls,
                usage=usage,
            )
        except Exception as e:
            print(f"❌ Failed to parse response: {e}")
            logger.error(f"Failed to parse response: {e}")
            raise

    async def close(self):
        await self._client.aclose()