from dataclasses import dataclass, field
from typing import List, Dict
import json


@dataclass
class WorkingMemory:
    original_query: str                                      # 初始任务
    finished_tasks: List[str] = field(default_factory=list)  # 子任务序列
    execution_log: List[str] = field(default_factory=list)   # 执行日志

    def add_log(self, step: int, tool: str, args: Dict, result: str, status: str):
        """
        记录完整的执行三元组: (Tool, Args, Result)
        """
        # 1. 为了防止 Log 过长，对参数做简化的序列化
        # 比如把很长的 GeoJSON 坐标截断，只保留关键路径和配置
        args_str = self._format_args(args)

        # 2. 截断 Result
        res_str = str(result)
        if len(res_str) > 200:
            res_str = res_str[:200] + "...(truncated)"

        log_entry = f"[Step {step}] Tool: {tool}\n    Args: {args_str}\n    Result ({status}): {res_str}"
        self.execution_log.append(log_entry)

    def _format_args(self, args: Dict) -> str:
        """辅助函数：美化并截断参数字符串"""
        clean_args = {}
        for k, v in args.items():
            v_str = str(v)
            # 如果某个参数值特别长截断它
            if len(v_str) > 100:
                clean_args[k] = v_str[:50] + "..."
            else:
                clean_args[k] = v
        return json.dumps(clean_args, ensure_ascii=False)

    def get_prompt_view(self) -> str:
        """生成给 LLM 看的压缩版记忆"""
        return json.dumps({
            "finished_tasks": self.finished_tasks,
            "recent_history": self.execution_log[-3:]
        }, indent=2)