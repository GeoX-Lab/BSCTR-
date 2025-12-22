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
        """给 Planner / Tool Selector / RePlanner 使用的状态快照"""
        view = {
            "goal": self.original_query,
            "current_task": self.current_task,
            "finished_tasks": self.finished_tasks,
            "available_artifacts": list(self.artifacts.keys()),
            "recent_steps": self.recent_steps[-3:]
        }
        return json.dumps(view, ensure_ascii=False, indent=2)

    def get_replan_context(self) -> str:

        completed = self.finished_tasks if self.finished_tasks else ["None"]
        artifact_lines = []
        if self.artifacts:
            for name, value in self.artifacts.items():
                artifact_lines.append(f"- {name}: {value}")
        else:
            artifact_lines.append("- None")

        tool_lines = []
        for tool, info in self.tool_context.items():
            if info.get("success"):
                result = info.get("last_result", "N/A")
                step = info.get("step", "N/A")
                tool_lines.append(
                    f"- Step {step}: Tool '{tool}' produced output: {result}"
                )

        if not tool_lines:
            tool_lines.append("- None")

        last_failure = None
        for step in reversed(self.recent_steps):
            if step.get("status") == "FAIL":
                last_failure = step
                break

        if last_failure:
            failure_block = (
                f"- Failed Task: {last_failure.get('current_task', 'Unknown')}\n"
                f"- Tool: {last_failure.get('tool', 'Unknown')}\n"
                f"- Error Type: {last_failure.get('error_type', 'Unknown')}\n"
                f"- Reason: {last_failure.get('reason', 'Unknown')}"
            )
        else:
            failure_block = "- None"

        context = f"""
            ### Successfully Completed Tasks
            {chr(10).join(f"- {t}" for t in completed)}
        
            ### Available Data Artifacts (Reusable)
            {chr(10).join(artifact_lines)}
        
            ### Available Tool Outputs (Reusable)
            {chr(10).join(tool_lines)}
        
            ### Last Failure
            {failure_block}
            """

        return context.strip()
