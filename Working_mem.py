from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json


@dataclass
class WorkingMemory:
    # === 全局目标 ===
    original_query: str

    # === 任务状态 ===
    finished_tasks: List[str] = field(default_factory=list)
    current_task: Optional[str] = None

    # === 全局归档 ===
    global_history: List[Dict[str, Any]] = field(default_factory=list)
    # === 工具执行上下文 ===
    tool_context: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # === 最近执行记录===
    recent_steps: List[Dict[str, Any]] = field(default_factory=list)

    def start_task(self, task_query: str):
        """
        开始一个新子任务：
        1. 将上一任务的 recent_steps 归档到 global_history
        2. 清空 recent_steps 以便为新任务提供干净的 Context
        3. 更新 current_task 名称
        """
        # 1. 归档旧记忆
        if self.recent_steps:
            self.global_history.extend(self.recent_steps)

        # 2. 设置新任务名
        self.current_task = task_query

    def record_tool_success(self, step: int, tool: str, args: Dict, result: str):

        step_record = {
            "current_task": self.current_task,
            "tool": tool,
            "args": args,
            "result": result,
            "status": "SUCCESS"
        }
        self.tool_context[str(step)] = step_record
        self.recent_steps.append(step_record)

    def record_tool_failure(self, step: int, tool: str, error_type: str, reason: str):

        step_record = {
            "current_task": self.current_task,
            "tool": tool,
            "error_type": error_type,
            'reason': reason,
            "status": "FAIL"
        }
        self.tool_context[str(step)] = step_record
        self.recent_steps.append(step_record)

    def add_feedback_message(self, message: str):
        self.recent_steps.append({
            "role": "system_feedback",
            "content": message
        })

    def get_final_report_view(self) -> str:
        """
        最终写报告时，需要看所有历史
        """
        full_history = self.global_history + self.recent_steps
        view = {
            "goal": self.original_query,
            "execution_log": full_history
        }
        return json.dumps(view, ensure_ascii=False, indent=2)
