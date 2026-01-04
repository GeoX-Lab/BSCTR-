from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json, copy


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
        最终写报告时，需要看所有历史。
        优化点：保留成功步骤 + 失败步骤（尤其是最后一步的报错），并截断过长内容。
        """
        # 1. 获取完整历史
        raw_history = self.global_history + self.recent_steps

        # 2. 如果历史为空，直接返回
        if not raw_history:
            return json.dumps({"goal": self.original_query, "execution_log": []})

        history_list = []

        # 3. 遍历处理
        for i, step in enumerate(raw_history):
            # 深度拷贝，防止修改原始数据
            clean_step = copy.deepcopy(step)
            is_last_step = (i == len(raw_history) - 1)

            if clean_step.get("status") == "SUCCESS" or is_last_step:

                # --- 截断逻辑 (保持你写的，很好) ---
                if "result" in clean_step and isinstance(clean_step["result"], str):
                    if len(clean_step["result"]) > 500:  # 稍微给多一点，500有时候太短看不出报错细节
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
