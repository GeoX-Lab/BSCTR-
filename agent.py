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
