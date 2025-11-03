import os
import json
from Toolregistry import ToolRegistry
from model import LLM
from ToolMem import ToolMem

class BaseAgent:
    def __init__(self,
                 initial_model: str,
                 sys_prompt_template: str,
                 output_dir: str = "outputs"):

        self.initial_model = initial_model
        self.sys_prompt_template = sys_prompt_template
        self.output_dir = output_dir
        self.llm = LLM(initial_model, sys_prompt_template)
        self.tool_registry = ToolRegistry()
        self.tool_mem = ToolMem()
        self.history = []
        self.tool = []

    def history_save(self, filename: str):
        if filename is None:
            filename = os.path.join(self.output_dir, "history.json")
        with open(filename, "w") as f:
            for chunk in self.history:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    async def call_tool(self, tool_name: str, arguments: dict):
        tool_result = ""
        try:
            if tool_name is not None and tool_name in self.tool:
                tool_result = self.tool_registry.run_tool(tool_name, arguments)
            yield tool_result

        except Exception as e:
            yield "Exception while running tool {}: {}".format(tool_name, e)

    async def chat(self, prompt, llm_name):
        if llm_name is not None:
            self.llm = LLM(llm_name, prompt)
        res = self.llm.generate_stream_res(prompt)
        return res

class Agent(BaseAgent):
    def __init__(self, initial_model: str, sys_prompt_template: str, output_dir: str = "outputs"):
        super().__init__(initial_model, sys_prompt_template, output_dir)
