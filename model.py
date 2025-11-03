import os
import yaml
from typing import Dict, Any, Optional, AsyncGenerator
from openai import AsyncOpenAI

def load_model_config_from_yaml(yaml_path: str, model: str) -> Dict[str, Any]:
    """
    从 YAML 文件加载模型配置。
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML 配置文件不存在: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if model not in data:
        raise KeyError(f"在 {yaml_path} 中未找到模型配置: {model}")
    cfg = data[model]
    return {
        "use_responses_api": bool(cfg.get("use_responses_api", True)),
        "client_kwargs": cfg.get("client_kwargs", {}),
        "generation": cfg.get("generation", {}),
    }

class LLM:
    """
    只用 AsyncOpenAI 的 LLM 封装，从 YAML 配置加载参数。
    """

    def __init__(self, yaml_path: str, model: str, sys_prompt: Optional[str] = None):
        """
        yaml_path: models.yaml 的路径
        model_key: YAML 中的 top-level key（例如 gpt4o_default）
        sys_prompt: 系统提示（可选）
        """
        self.yaml_path = yaml_path
        self.model = model
        self.sys_prompt = sys_prompt
        self.config = load_model_config_from_yaml(yaml_path, model)
        client_kwargs = dict(self.config.get("client_kwargs"))
        self.client = AsyncOpenAI(**client_kwargs)

    async def generate_stream_res(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        异步生成器
        """
        gen_conf = self.config.get("generation", {})
        max_tokens = max_tokens if max_tokens is not None else gen_conf.get("max_tokens")
        temperature = temperature if temperature is not None else gen_conf.get("temperature")

        client = self.client
        if client is None:
            yield {"type": "error", "error": "OpenAI 客户端未初始化。"}
            return

        use_responses_api = bool(self.config.get("use_responses_api", True))
        model_id = self.config.get("model_id")

        try:
            if use_responses_api:
                input_payload = (self.sys_prompt + "\n" + prompt) if self.sys_prompt else prompt
                call_kwargs = dict(
                    model=model_id,
                    input=input_payload,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                call_kwargs.update(kwargs)
                resp_or_iter = await client.responses.create(**call_kwargs)

                # 流式：异步可迭代对象
                if hasattr(resp_or_iter, "__aiter__"):
                    async for chunk in resp_or_iter:
                        text_piece = getattr(chunk, "output_text", None)
                        if text_piece is None and isinstance(chunk, dict):
                            if "output_text" in chunk:
                                text_piece = chunk["output_text"]
                            elif "choices" in chunk and chunk["choices"]:
                                delta = chunk["choices"][0].get("delta")
                                text_piece = delta.get("content") if delta else None
                        if text_piece:
                            yield text_piece
                else:
                    text = getattr(resp_or_iter, "output_text", None) or str(resp_or_iter)
                    yield text
        except Exception as e:
            yield {"error": f"调用 OpenAI API 失败: {e}"}
            return

    async def generate_res(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:

        gen_conf = self.config.get("generation", {})
        max_tokens = max_tokens if max_tokens is not None else gen_conf.get("max_tokens")
        temperature = temperature if temperature is not None else gen_conf.get("temperature")

        client = self.client
        if client is None:
            return "OpenAI 客户端未初始化。"

        use_responses_api = bool(self.config.get("use_responses_api", True))
        model_id = self.config.get("model_id")

        try:
            if use_responses_api:
                input_payload = (self.sys_prompt + "\n" + prompt) if self.sys_prompt else prompt
                call_kwargs = dict(
                    model=model_id,
                    input=input_payload,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                call_kwargs.update(kwargs)

                resp = await client.responses.create(**call_kwargs)
                text = getattr(resp, "output_text", None) or (
                    resp.get("output_text") if isinstance(resp, dict) else str(resp))

                return text

        except Exception as e:
            return f"error {e}"
