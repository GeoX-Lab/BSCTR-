from typing import Callable, Dict, Any, get_origin
import json
import importlib
from fastmcp import FastMCP
import inspect


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

        with open(path, 'r', encoding='utf-8') as f:
            tools_json = json.load(f)
        for tool in tools_json:
            tool_name = tool.get("name")
            tool_callable = tool.get("callable")
            meta = tool.get("meta", {})
            if tool_name and tool_callable:
                if isinstance(tool_callable, str):
                    tool_callable = self._resolve_callable(tool_callable)
                self.register_tool(tool_name, tool_callable, meta)

        return tools_json

    def get_tool(self, tool_name: str):

        if tool_name in self.tools:
            return self.tools.get(tool_name)
        else:
            return None

    def register_tool(self, tool_name: str, tool_callable: Callable, meta: dict):

        if not isinstance(meta, dict):
            return "Meta must be a dictionary!"
        self.tools[tool_name] = {"callable": tool_callable, "meta": meta}

    def _infer_parameters_from_callable(self, fn: Callable) -> Dict[str, Any]:

        sig = inspect.signature(fn)

        properties = {}
        required = []

        for name, param in sig.parameters.items():
            ann = param.annotation
            json_type = "string"

            if ann in (int, float):
                json_type = "number"
            elif ann is bool:
                json_type = "boolean"
            elif ann is list:
                json_type = "array"
            else:
                origin = get_origin(ann)
                if origin in (list, tuple):
                    json_type = "array"

            properties[name] = {
                "type": json_type,
                "description": ""
            }

            if param.default is inspect._empty:
                required.append(name)

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }

    def extract_short_description(self, raw_desc: str) -> str:

        if not raw_desc:
            return ""

        cleaned_desc = raw_desc.strip().replace("\n", " ").replace("\r", "")

        param_index = cleaned_desc.lower().find("parameters")

        if param_index != -1:
            return cleaned_desc[:param_index].strip()

        return cleaned_desc

    def load_from_fastmcp(self, mcp: FastMCP):

        print(f"[*] Loading tools from FastMCP: {mcp.name}...")

        tools: dict = mcp._tool_manager._tools
        for tool_name, tool_obj in tools.items():

            tool_callable = (
                    getattr(tool_obj, "fn", None)
                    or getattr(tool_obj, "callable", None)
            )

            if not callable(tool_callable):
                print(f"[!] Skip tool '{tool_name}': not callable")
                continue

            inferred_parameters = self._infer_parameters_from_callable(tool_callable)
            raw_desc = getattr(tool_obj, "description", "") or ""
            meta = {
                "description": self.extract_short_description(raw_desc),
                "parameters": inferred_parameters,
                "source": "fastmcp",
            }

            self.register_tool(tool_name, tool_callable, meta)
            print(f"    - Registered MCP tool: {tool_name}")

    def generate_tool_schema(self, tool_name: str) -> dict:

        tool = self.get_tool(tool_name)
        if not tool:
            raise KeyError(f"Tool '{tool_name}' not found in registry")

        meta = tool.get("meta", {})
        params_meta: Dict[str, Any] = meta.get("parameters", {}) or {}

        # 如果已经是完整的 schema 格式（包含 type 和 properties），则直接返回
        if "properties" in params_meta and isinstance(params_meta.get("properties"), dict):
            return params_meta
        
        # 否则，构建 schema
        properties, required = {}, []
        for k, spec in params_meta.items():
            if isinstance(spec, dict):
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

        tool = self.get_tool(tool_name)
        if not tool:
            raise KeyError(f"Tool '{tool_name}' not found")

        meta = tool.get("meta", {})
        
        # 如果 meta 中没有 parameters 或为空，则从 callable 推断
        if "parameters" not in meta or not meta.get("parameters"):
            try:
                fn = tool.get("callable")
                if fn and callable(fn):
                    schema = self._infer_parameters_from_callable(fn)
                else:
                    schema = {}
            except Exception:
                schema = {}
        else:
            try:
                schema = self.generate_tool_schema(tool_name)
            except Exception:
                schema = meta.get("parameters", {})

        desc = meta.get("description") or tool.get("description") or tool_name

        return {
            "name": tool_name,
            "description": desc.strip(),
            "parameters": schema
        }
