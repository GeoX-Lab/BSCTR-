from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json, copy


@dataclass
class WorkingMemory:

    original_query: str
    current_task: Optional[str] = None
    finished_tasks: List[str] = field(default_factory=list)
    global_history: List[Dict[str, Any]] = field(default_factory=list)
    recent_steps: List[Dict[str, Any]] = field(default_factory=list)
    feedback_message: List[Dict[str, Any]] = field(default_factory=list)

    def start_task(self, task_query: str):

        if self.recent_steps:
            self.global_history.extend(self.recent_steps)
        self.current_task = task_query

    def record_step(self, tool: str, args: Dict, result: str, outcome):

        step_record = {
            "current_task": self.current_task,
            "tool": tool,
            "args": args,
            "result": result,
            "outcome": outcome
        }
        self.recent_steps.append(step_record)

    def add_feedback_message(self, message: str):
        self.feedback_message.append({
            "role": "Failed_feedback",
            "content": message
        })

    def get_final_report_view(self) -> str:
        """
        总结历史记录
        """
        raw_history = self.global_history + self.recent_steps

        if not raw_history:
            return json.dumps({"goal": self.original_query, "execution_log": []})

        history_list = []

        for i, step in enumerate(raw_history):
            clean_step = copy.deepcopy(step)
            is_last_step = (i == len(raw_history) - 1)

            if clean_step.get("status") == "SUCCESS" or is_last_step:
                if "result" in clean_step and isinstance(clean_step["result"], str):
                    if len(clean_step["result"]) > 500: 
                        clean_step["result"] = clean_step["result"][:800] + "... [Truncated]"

                if "args" in clean_step:
                    args_str = str(clean_step["args"])
                    if len(args_str) > 500:
                        clean_step["args"] = args_str[:500] + "... [Truncated]"

                history_list.append(clean_step)

        view = {
            "goal": self.original_query,
            "execution_log": history_list
        }
        return json.dumps(view, ensure_ascii=False, indent=2)
