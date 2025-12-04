import json
from typing import Any, Optional
import inspect
from model import LLM
from Toolregistry import ToolRegistry
class BaseAgent:
    def __init__(self, initial_model: str, sys_prompt_template: str, output_dir: str = "outputs"):

        self.initial_model = initial_model
        self.sys_prompt_template = sys_prompt_template
        self.output_dir = output_dir
        self.llm = LLM(initial_model)

        self.tool_registry = ToolRegistry()
        self.tool_list = []
        self.history = []

    def _normalize_chunk(self, chunk: Any) -> str:
        if isinstance(chunk, dict):
            if chunk.get("type") == "error":
                raise RuntimeError(chunk.get("error", "unknown tool error"))
            return chunk.get("text", json.dumps(chunk, ensure_ascii=False))
        if isinstance(chunk, (bytes, bytearray)):
            return chunk.decode("utf-8", errors="ignore")
        return str(chunk)

    async def call_tool(self, tool_name: str, arguments: Optional[dict] = None):
        arguments = arguments or {}
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return f"Tool '{tool_name}' not found."

        fn = tool["callable"]

        try:
            out = fn(**arguments)
            if inspect.isawaitable(out):
                out = await out

            if hasattr(out, "__aiter__"):
                acc = ""
                async for chunk in out:
                    acc += self._normalize_chunk(chunk)
                text = acc

            elif hasattr(out, "__iter__") and not isinstance(out, (str, bytes, dict)):
                acc = ""
                for chunk in out:
                    acc += self._normalize_chunk(chunk)
                text = acc

            else:
                text = self._normalize_chunk(out)

            # 记录工具调用轨迹
            self.tool_list.append(tool_name)

            # 写入历史（若你未来需要 ReAct，可用到）
            self.history.append({
                "role": "tool",
                "name": tool_name,
                "content": text
            })

            return text

        except Exception as e:
            return f"Exception calling tool {tool_name}: {e}"

    async def chat(self, prompt, llm_name=None, image_path=None):

        if llm_name:
            self.llm = LLM(llm_name)

        # 写入用户消息（保持最简单历史）
        self.history.append({"role": "user", "content": prompt})

        # 流式输出
        result = None
        acc = []

        async for chunk in self.llm.generate_stream_res(
            prompt=prompt,
            history=self.history,
            image_path=image_path
        ):
            if chunk.get("type") == "error":
                return f"Error: {chunk.get('error')}"

            if chunk.get("type") in ("text", "stream"):
                acc.append(chunk.get("text", ""))

            if chunk.get("type") == "final":
                result = chunk.get("text", "")

        result = result if result is not None else "".join(acc)

        self.history.append({"role": "assistant", "content": result})

        return result

class JudgeAgent:
    """
    使用 LLM 判断任务是否完成：
    - 输入：用户原始任务 + ExecutorAgent 的输出内容
    - 输出： {"done": true/false, "reason": "..."}
    """

    def __init__(self, model_name: str, judge_prompt: str):
        self.llm = LLM(model_name)
        self.judge_prompt = judge_prompt

    async def judge(self, user_task: str, executor_output: str) -> bool:
        """
        返回 True/False
        """
        full_prompt = self.judge_prompt.format(
            user_task=user_task,
            executor_output=executor_output
        )

        res = await self.llm.generate_res(full_prompt)
        res = res.lower()

        if "yes" in res or "true" in res:
            return True

        return False
