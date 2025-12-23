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

    # === 工具执行上下文 ===
    tool_context: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # === 智能体中间产物 ===
    artifacts: Dict[str, Any] = field(default_factory=dict)

    # === 最近执行记录===
    recent_steps: List[Dict[str, Any]] = field(default_factory=list)

    def start_task(self, task_query: str):
        self.current_task = task_query

    def record_tool_success(self, step: int, tool: str, args: Dict, result_summary: str):
        self.tool_context[tool] = {
            "step": step,
            "last_args": args,
            "last_result": result_summary,
            "success": True
        }
        self.recent_steps.append({
            "tool": tool,
            "status": "SUCCESS"
        })

    def record_tool_failure(self, current_task: str, tool: str, error_type: str, reason: str):
        self.recent_steps.append({
            "current_task": current_task,
            "tool": tool,
            "status": "FAIL",
            "error_type": error_type,
            'reason': reason
        })
    # ---------- 给 LLM 的视图 ----------

    def get_prompt_view(self) -> str:

        view = {
            "goal": self.original_query,
            "current_task": self.current_task,
            "finished_tasks": self.finished_tasks,
            "tool_context": self.tool_context,
        }
        return json.dumps(view, ensure_ascii=False, indent=2)

    def get_error_history(self, task_query: str) -> str:
        """
        从 recent_steps 中筛选出属于当前 task_query 的失败记录，
        并格式化为字符串，用于 Prompt 提示。
        """
        relevant_failures = [
            step for step in self.recent_steps
            if step.get("status") == "FAIL" and step.get("current_task") == task_query
        ]

        if not relevant_failures:
            return "None (No previous failures for this step)."

        history_lines = []
        for i, f in enumerate(relevant_failures):
            history_lines.append(f"--- Failure Record {i + 1} ---")
            history_lines.append(f"Target Tool: {f.get('tool')}")
            history_lines.append(f"Error Type:  {f.get('error_type')}")
            history_lines.append(f"Detailed Reason: {f.get('reason')}")
            history_lines.append("")

        return "\n".join(history_lines).strip()
