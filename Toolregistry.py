from typing import Callable, Dict, Any
class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def load_tool(self):
        pass

    def get_tool(self, tool_name):
        if tool_name in self.tools:
            return self.tools.get(tool_name)
        else:
            return "Tool is not found in registry!"

    def register_tool(self, tool_name: str, tool_callable: Callable, meta: dict):
        if not isinstance(meta, dict):
            return "Meta must be a dictionary!"
        self.tools[tool_name] = {"callable": tool_callable, "meta": meta}

    def generate_tool_schema(self, tool_name, arguments: dict):
        """
        {
            "name": "tool_name",
            "description": "...",
            "parameters": {
                "arg1": {"type":"str","required":True, "example": "..."},
                ...
            }
        }
        """
        tool = self.get_tool(tool_name)
        if tool is None:
            raise KeyError(f"Tool '{tool_name}' not found in registry")

        meta = tool.get("meta", {})
        schema = {
            "name": tool_name,
            "description": meta.get("description", ""),
            "parameters": {}
        }

        if "parameters" in meta and isinstance(meta["parameters"], dict):
            schema["parameters"] = meta["parameters"]
            return schema

        if arguments and isinstance(arguments, dict):
            for k, v in arguments.items():
                schema["parameters"][k] = {
                    "type": type(v).__name__,
                    "required": v is not None,
                    "example": v
                }
        return schema
