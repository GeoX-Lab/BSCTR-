import os
import json
import inspect
from typing import Any, Optional, List, Dict
from model import LLM
from Toolregistry import ToolRegistry


class BaseAgent:
    def __init__(self, initial_model: str, sys_prompt_template: str, output_dir: str):

        self.initial_model = initial_model
        self.sys_prompt_template = sys_prompt_template
        self.output_dir = output_dir
        self.llm = LLM(initial_model)

        self.tool_registry = ToolRegistry()
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

            self.history.append({
                "role": "tool",
                "name": tool_name,
                "content": text
            })

            return True, text

        except Exception as e:
            error = f"Exception calling tool {tool_name}: {e}"
            return False, error
    
    async def _call_llm(self, prompt: str, history: List[Dict] = None) -> str:
        """
        调用LLM接口
        """
        text_pieces = []
        final_text = None

        async for chunk in self.llm.generate_stream_res(prompt=prompt, history=history):
            c_type = chunk.get("type")
            raw_text = chunk.get("text")
            c_text = raw_text if raw_text is not None else ""
            
            if c_type in ("text", "stream"):
                text_pieces.append(c_text)
            
            elif c_type == "final":
                final_text = c_text

        if final_text:
            return final_text.strip()
        else:
            return "".join([t for t in text_pieces if t is not None]).strip()
        
    def _save_tool(self, save_top_k: int = 20):
        """
        根据分数排序，将 Top-K 工具名称保存到 JSON 文件
        """
        all_tools = list(self.tool_set.values())

        all_tools.sort(key=lambda x: x.get("score", 0), reverse=True)

        top_tools = all_tools[:save_top_k]
        current_tool_names = [t["name"] for t in top_tools]

        if os.path.exists(self.tool_dir):
            try:
                with open(self.tool_dir, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

        existing_indices = [int(k) for k in data.keys() if k.isdigit()]
        if existing_indices:
            new_index = str(max(existing_indices) + 1)
        else:
            new_index = "0"

        data[new_index] = current_tool_names

        with open(self.tool_dir, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f">>> [Log] Saved Top-{len(current_tool_names)} tools (from pool of {len(all_tools)}) to {self.tool_dir}")
        print(f"    Selected tools: {current_tool_names}")

    def save_data(self, query: str, final_result: str, id: str, status: str = "finished"):
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        safe_filename = str(id).replace("/", "_").replace("\\", "_")
        file_path = os.path.join(self.output_dir, f"{safe_filename}.json")

        data = {
            "task_id": id,
            "status": status,
            "query": query,
            "final_result": final_result,
            "history": list(self.history),
        }
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"[*] Task {id} archived to {file_path}")
        except Exception as e:
            print(f"[!] Failed to save task {id}: {e}")
