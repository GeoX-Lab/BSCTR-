from typing import Callable, Dict, Any
import json
import importlib
from fastmcp import FastMCP

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _resolve_callable(dotted: str):
        sep = ":" if ":" in dotted else "."
        mod_path, func_name = dotted.rsplit(sep, 1)
        module = importlib.import_module(mod_path)
        fn = getattr(module, func_name)
        if not callable(fn):
            raise TypeError(f"{dotted} is not callable")
        return fn

    def load_tool(self, path: str):
        """
        从文件加载工具数据，并注册工具。
        """
        with open(path, 'r', encoding='utf-8') as f:
            tools_json = json.load(f)
        for tool in tools_json:
            # 假设 JSON 文件中的每个工具都包含 'name', 'callable', 'meta'
            tool_name = tool.get("name")
            tool_callable = tool.get("callable")
            meta = tool.get("meta", {})
            if tool_name and tool_callable:
                if isinstance(tool_callable, str):
                    tool_callable = self._resolve_callable(tool_callable)
                self.register_tool(tool_name, tool_callable, meta)

        return tools_json

    def get_tool(self, tool_name: str):
        """
        获取指定工具的信息。
        """
        if tool_name in self.tools:
            return self.tools.get(tool_name)
        else:
            return None  # 返回 None 而不是字符串，便于后续处理

    def register_tool(self, tool_name: str, tool_callable: Callable, meta: dict):
        """
        注册一个工具。
        """
        if not isinstance(meta, dict):
            return "Meta must be a dictionary!"
        self.tools[tool_name] = {"callable": tool_callable, "meta": meta}

    def load_from_fastmcp(self, mcp: FastMCP):
        """
        解析 FastMCP 对象，将所有 @mcp.tool 注册到当前 Registry。
        """
        print(f"[*] Loading tools from FastMCP: {mcp.name}...")

        # 遍历 FastMCP 内部存储的工具字典
        # FastMCP 的 _tools 属性存储了所有注册的工具定义
        for name, tool_def in mcp._tools.items():
            # 1. 获取函数本体 (Executable)
            # tool_def.fn 就是那个被装饰的 Python 原生函数
            tool_fn = tool_def.fn

            # 2. 获取元数据 (Metadata)
            # FastMCP 已经自动生成了标准的 parameters schema (OpenAI 格式)
            schema = tool_def.parameters_schema
            description = tool_def.description or ""

            # 3. 构建 meta 字典 (适配你的 get_unified_tool_info 逻辑)
            # 你的代码逻辑是优先读取 meta['parameters']，所以直接存进去即可
            meta = {
                "description": description,
                "parameters": schema
            }

            # 4. 注册
            self.register_tool(name, tool_fn, meta)
            print(f"    - Registered MCP tool: {name}")

    def generate_tool_schema(self, tool_name: str) -> dict:
        """
        生成工具的 schema，供 LLM 解析使用。
        """
        tool = self.get_tool(tool_name)
        if not tool:
            raise KeyError(f"Tool '{tool_name}' not found in registry")

        meta = tool.get("meta", {})
        params_meta: Dict[str, Any] = meta.get("parameters", {}) or {}

        properties, required = {}, []
        for k, spec in params_meta.items():
            # 期望 spec 里至少含有 type/description/required
            properties[k] = {
                "type": spec.get("type", "string"),
                "description": spec.get("description", "")
            }
            if "enum" in spec:
                properties[k]["enum"] = spec["enum"]
            if spec.get("required", False):
                required.append(k)

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    def get_unified_tool_info(self, tool_name: str) -> dict:
        """
        返回一个完全统一、结构化的工具信息，
        包含 embedding 和 LLM 都能一致看到的字段。

        返回结构：
        {
            "name": "...",
            "description": "...",
            "parameters": {...schema...},
            "meta": {...原始meta...}
        }
        """
        tool = self.get_tool(tool_name)
        if not tool:
            raise KeyError(f"Tool '{tool_name}' not found")

        meta = tool.get("meta", {})
        if "parameters" in meta and meta["parameters"]:
            schema = meta["parameters"]
        else:
            # 否则尝试根据函数签名生成 (兼容旧逻辑)
            try:
                schema = self.generate_tool_schema(tool_name)
            except Exception:
                schema = {}

            # 描述优先取 meta 里的
        desc = meta.get("description") or tool.get("description") or tool_name

        return {
            "name": tool_name,
            "description": desc.strip(),
            "parameters": schema,
            "meta": meta
        }
