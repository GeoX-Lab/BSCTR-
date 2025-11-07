import os
import json
import time
from typing import Any, Optional, Dict, List
import inspect
from datetime import datetime
from model import LLM
from ToolMem import ToolMem
from Toolregistry import ToolRegistry
from prompt import tool_agent_prompt

class BaseAgent:
    def __init__(self,
                 initial_model: str,
                 sys_prompt_template: str,
                 output_dir: str = "outputs"):

        self.initial_model = initial_model
        self.sys_prompt_template = sys_prompt_template
        self.output_dir = output_dir
        self.llm = LLM(initial_model)
        self.tool_registry = ToolRegistry()
        self.tool_mem = ToolMem()
        self.history = []
        self.tool_list = []

    def history_save(self, filename: str):
        if filename is None:
            filename = os.path.join(self.output_dir, "history.json")
        with open(filename, "w") as f:
            for chunk in self.history:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    @staticmethod
    def _get_timestamp(format_type: str = "default") -> str:
        """
        获取当前时间戳
        """
        now = datetime.now()

        if format_type == "iso":
            return now.isoformat()
        elif format_type == "compact":
            return now.strftime("%Y%m%d_%H%M%S")
        elif format_type == "timestamp":
            return str(int(time.time()))
        else:  # default
            return now.strftime("%Y-%m-%d %H:%M:%S")

    def _append_msg(self, role: str, text: str):
        self.history.append({
            "role": role,
            "content": [{"type": "text", "text": text}],
            "ts": self._get_timestamp("iso")
        })

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
            return f"Tool '{tool_name}' not found in the registry."

        fn = tool["callable"]
        try:
            out = fn(**arguments)
            # awaitable（协程对象）
            if inspect.isawaitable(out):
                out = await out
            # 异步可迭代（async generator / async iterable）
            if hasattr(out, "__aiter__"):
                acc = ""
                async for chunk in out:
                    acc += self._normalize_chunk(chunk)
                text = acc
            # 同步可迭代（generator/iterable，但非字符串/字节/dict）
            elif hasattr(out, "__iter__") and not isinstance(out, (str, bytes, dict)):
                acc = ""
                for chunk in out:
                    acc += self._normalize_chunk(chunk)
                text = acc
            else:
                text = self._normalize_chunk(out)

            self.history.append({
                "role": "tool",
                "name": tool_name,
                "arguments": arguments,
                "content": [{"type": "text", "text": text}],
                "ts": self._get_timestamp("iso")
            })
            return text

        except Exception as e:
            err = f"Exception while running tool {tool_name}: {e}"
            self.history.append({
                "role": "tool",
                "name": tool_name,
                "arguments": arguments,
                "content": [{"type": "error", "text": str(e)}],
                "ts": self._get_timestamp("iso")
            })
            return err

    def _ensure_system_in_history(self):
        """
        将 sys_prompt_template写入history
        """
        if not self.history or self.history[0].get("role") != "system":
            self.history.insert(0, {"role": "system", "text": self.sys_prompt_template})
    async def chat(self, prompt, llm_name, image_path = None):
        """
        带有历史记录的对话函数
        """
        if llm_name:
            self.llm = LLM(llm_name)
        self._ensure_system_in_history()

        result = None
        acc = []
        if llm_name is not None:
            self.llm = LLM(llm_name)
        async for chunk in self.llm.generate_stream_res(prompt, self.history, image_path):
            if chunk.get("type") == "error":
                return f"Error: {chunk.get('error')}"
            if chunk.get("type") in ("text", "stream"):
                 acc.append(chunk.get("text", ""))
            if chunk.get("type") == "final":
                result = chunk.get("text", "")

        result = result if result is not None else "".join(acc)
        self._append_msg("user", prompt)
        self._append_msg("text", result)
        return result

class ToolAgent(BaseAgent):
    """
    ToolAgent：
    """
    def __init__(self, initial_model: str, sys_prompt_template: str, output_dir: str = "outputs"):
        super().__init__(initial_model, sys_prompt_template, output_dir)
        self.tool_history: List[Dict[str, Any]] = []

    def retrieve_tools(self, task: str) -> List[Dict[str, Any]]:
        """从 ToolMem 获取候选工具集合"""
        return self.tool_mem.get_similar_tools(task)

    async def execute_decided_tool(self, tool_decision: Dict[str, Any]):
        """执行 LLM 决策出的工具，并返回执行结果"""
        if not tool_decision or "tool_name" not in tool_decision:
            return "No valid tool decision."

        tool_name = tool_decision["tool_name"]
        arguments = tool_decision.get("arguments", {})

        result = await self.call_tool(tool_name, arguments)

        # 记录执行历史
        self.history.append({
            "role": "tool",
            "name": tool_name,
            "arguments": arguments,
            "content": [{"type": "text", "text": result}],
            "feedback": "Executed by ToolAgent decision",
            "ts": self._get_timestamp("iso")
        })

        self._extract_tool_history()
        return result

    async def execute_task(self, instruction: str) -> Dict[str, Any]:
        """
        处理单个任务：
        1) 检索候选工具
        2) 让 LLM 决策（decide_tool）
        3) 执行被选工具（execute_decided_tool）
        """
        candidates = self.retrieve_tools(instruction)
        if not candidates:
            return {"task": instruction, "result": "No tools found"}

        tool_list_str = json.dumps([t["name"] for t in candidates], ensure_ascii=False)
        prompt = tool_agent_prompt.format(tool_list=tool_list_str)
        llm_response = await self.chat(prompt, self.initial_model)
        tool_choice = self._parse_llm_response(llm_response)
        result = await self.execute_decided_tool(tool_choice)

        return {
            "task": instruction,
            "tool_choice": tool_choice,
            "result": result
        }

    async def execute_tasks(self, tasks: List[Dict[str, Any]]):
        """
        """
        all_results = []
        for task in tasks:
            instruction = task.get("instruction", "")
            try:
                out = await self.execute_task(instruction)
                all_results.append(out)
            except Exception as e:
                all_results.append({"task": instruction, "error": str(e)})
        return all_results

    def _extract_tool_history(self):
        """从历史中提取工具使用记录"""
        self.tool_history = [h for h in self.history if h.get("role") == "tool"]
    @staticmethod
    def _parse_llm_response(response: str) -> Optional[Dict[str, Any]]:
        """解析 LLM 输出中的工具选择结果"""
        try:
            response = response.strip()
            if response.startswith("{"):
                return json.loads(response)
            for line in response.splitlines():
                if "tool_name" in line:
                    return json.loads(line)
        except Exception as e:
            print(f"Failed to parse tool decision: {e}")
        return None

class ToolManager(BaseAgent):
    def __init__(self, initial_model: str, sys_prompt_template: str, output_dir: str = "outputs"):
        super().__init__(initial_model, sys_prompt_template, output_dir)
        self.initial_model = initial_model
        self.sys_prompt_template = sys_prompt_template

    def tool_manager(self, prompt, llm_name=None):
        self.llm = LLM(llm_name, self.sys_prompt_template)
        res = self.llm.generate_res(prompt)
        return res

    @staticmethod
    def evaluate_feedback(result) -> str:
        """
        根据任务执行结果评估工具节点的反馈
        例如，根据执行结果返回描述性反馈：“任务执行成功”、“任务执行失败”或“结果不明确”
        """
        if isinstance(result, str) and "error" in result.lower():
            # 如果执行结果包含 "error"，说明执行失败，返回失败的描述
            return "Task execution failed due to an error."
        elif result:
            # 如果执行结果非空且无错误，说明执行成功，返回成功的描述
            return "Task executed successfully."
        else:
            # 如果执行结果为空或不明确，返回结果不明确的描述
            return "Task executed with unclear results."
    @staticmethod
    def _prepare_arguments(sig: inspect.Signature, arguments: dict) -> inspect.BoundArguments:
        """准备函数参数，处理参数验证和默认值"""
        try:
            return sig.bind(**arguments)
        except TypeError as e:
            # 处理参数不匹配的情况
            raise ValueError(f"Argument mismatch for function: {str(e)}")

    @staticmethod
    def _parse_llm_response(llm_response: str) -> Optional[Dict[str, Any]]:
        """
        解析LLM响应，提取工具调用信息
        期望的格式示例:
        {
            "tool_name": "add_memory",
            "arguments": {
                "key": "user_preference",
                "value": "dark_mode"
            }
        }
        或者:
        TOOL: add_memory
        ARGUMENTS: {"key": "user_preference", "value": "dark_mode"}
        """
        try:
            # 尝试解析JSON格式
            if llm_response.strip().startswith('{'):
                return json.loads(llm_response)

            # 尝试解析键值对格式
            lines = llm_response.strip().split('\n')
            tool_info = {}

            for line in lines:
                if line.startswith('TOOL:'):
                    tool_info['tool_name'] = line.replace('TOOL:', '').strip()
                elif line.startswith('ARGUMENTS:'):
                    args_str = line.replace('ARGUMENTS:', '').strip()
                    tool_info['arguments'] = json.loads(args_str)

            return tool_info if tool_info else None

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse LLM response: {e}")
            return None

    def get_available_tools(self) -> list:
        """获取所有可用的工具列表"""
        tools = []
        for name in dir(self.tool_mem):
            if not name.startswith('_') and callable(getattr(self.tool_mem, name)):
                tools.append({
                    'name': name,
                    'doc': getattr(self.tool_mem, name).__doc__ or 'No documentation'
                })
        return tools

    def get_tool_history(self) -> list:
        """获取工具调用历史"""
        return self.history
