import os
import yaml
import base64
import aiofiles
from pathlib import Path
from typing import Dict, Any, Optional, AsyncGenerator, Union, List
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
    def __init__(self, model: str):
        """
        封装 LLM类，可实现 LLM与 VLM的流式与非流式输出
        流式输出格式：
        {"type": "final", "text": "result..."}
        非流式输出格式：
        "result..."

        Argument:
            model: YAML文件中的 model_name
        """
        self.yaml_path = "/media/csudxy0218/ZL/AgentToolmem/config.yaml"
        self.model = model
        self.config = load_model_config_from_yaml(self.yaml_path, model)
        client_kwargs = dict(self.config.get("client_kwargs"))
        self.client = AsyncOpenAI(**client_kwargs)

    async def generate_stream_res(
            self,
            prompt: str,
            history: List[Dict] = None,
            image_path = None,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        异步生成器，返回每一块文本输出。
        """
        gen_conf = self.config.get("generation", {})
        max_tokens = max_tokens if max_tokens is not None else gen_conf.get("max_tokens")
        temperature = temperature if temperature is not None else gen_conf.get("temperature")

        client = self.client
        if client is None:
            yield {"type": "error", "error": "OpenAI 客户端未初始化。"}
            return

        try:
            messages = await self.prepare_messages(prompt, image_path, history)
            call_kwargs = dict(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream = True
            )
            call_kwargs.update(kwargs)
            stream = await client.chat.completions.create(**call_kwargs)

            res: List[str] = []
            async for chunk in stream:
                piece = chunk.choices[0].delta.content
                if piece:
                    res.append(piece)
                yield {"type": "text", "text": piece}
            yield {"type": "final", "text": "".join(res)}

        except Exception as e:
            yield {"type": "error", "error": f"调用 OpenAI API 失败: {e}"}

    async def generate_res(
        self,
        prompt: str,
        history: List[Dict] = None,
        image_path = None,
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

        try:
            messages = await self.prepare_messages(prompt, image_path, history)
            call_kwargs = dict(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            call_kwargs.update(kwargs)
            res = await client.chat.completions.create(**call_kwargs)

            try:
                return res.choices[0].message.content or ""

            except Exception:
                if isinstance(res, dict):
                    ch = res.get("choices", [{}])[0]
                    msg = ch.get("message", {})
                    return msg.get("content", "") or str(res)
                return str(res)

        except Exception as e:
            return f"error {e}"

    async def prepare_messages(self, prompt: str, image_path: str, history: List[Dict]):

        messages = history.copy() if history is not None else []
        if image_path:
            base64_image = await self.image_to_base64(image_path)
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ]
        else:
            content = [{"type": "text", "text": prompt}]
        messages.append(
            {"role": "user", "content": content}
        )
        return messages

    @staticmethod
    async def image_to_base64(image_path: Union[str, Path]) -> str:
        async with aiofiles.open(image_path, "rb") as image_file:
            content = await image_file.read()
            encoded_string = base64.b64encode(content).decode("utf-8")
        return encoded_string
