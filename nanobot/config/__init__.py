"""Configuration for nanobot-lite."""

import json
from pydantic import BaseModel
from pathlib import Path

class Config(BaseModel):
    """Root configuration - simple and flat."""

    # 使用阿里模型
    model: str = "qwen3.5-plus"

    # DashScope Coding Plan OpenAI兼容接口
    api_base: str = "https://coding-intl.dashscope.aliyuncs.com/v1"

    # 你的 API Key
    api_key: str = "sk-sp-4ac5364922e1418d93712805d51d4c67"

    workspace: str = "~/.nanobot/workspace"
    max_iterations: int = 20
    temperature: float = 0.7

    # 建议稍微调大一点
    max_tokens: int = 4500

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace).expanduser()


CONFIG_PATH = Path.home() / ".nanobot" / "config.json"


def _camel_to_snake(name: str) -> str:
    result = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            result.append("_")
        result.append(ch.lower())
    return "".join(result)


def _convert_keys(data: dict) -> dict:
    return {_camel_to_snake(k): v for k, v in data.items()}


def load_config() -> Config:
    # if CONFIG_PATH.exists():
    #     try:
    #         data = json.loads(CONFIG_PATH.read_text())
    #         return Config.model_validate(_convert_keys(data))
    #     except Exception as e:
    #         print(f"Warning: Failed to load config: {e}, using defaults")
    return Config()


def save_config(config: Config) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Save in camelCase for readability
    data = config.model_dump()
    camel = {}
    for k, v in data.items():
        parts = k.split("_")
        camel_key = parts[0] + "".join(p.title() for p in parts[1:])
        camel[camel_key] = v
    CONFIG_PATH.write_text(json.dumps(camel, indent=2, ensure_ascii=False))
