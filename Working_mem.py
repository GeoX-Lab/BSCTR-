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
    # tool_name -> {last_args, last_result_summary, success}

    # === 中间产物 ===
    artifacts: Dict[str, Any] = field(default_factory=dict)
    # e.g. {"ndvi_raster": "...", "roi_mask": "..."}

    # === 最近执行记录（压缩态）===
    recent_steps: List[Dict[str, Any]] = field(default_factory=list)

    # ---------- 更新接口 ----------

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
