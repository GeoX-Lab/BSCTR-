import json
import requests
import yaml
import torch
from Agents.Base import BaseAgent
from collections import OrderedDict
from typing import List, Dict
from Prompt.geoplanbench import DECOMPOSE_PROMPT, ACT_PROMPT, SYSTEM_PROMPT, REPLAN_PROMPT
# from Prompt.apibank import DECOMPOSE_PROMPT, ACT_PROMPT, SYSTEM_PROMPT, REPLAN_PROMPT


class ToolAgent(BaseAgent):
    def __init__(self,
                 initial_model: str,
                 output_dir: str,
                 tool_dir: str,
                 yaml_path: str = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        self.output_dir = output_dir
        self.yaml_path = yaml_path
        super().__init__(initial_model, SYSTEM_PROMPT, self.output_dir)
        self.device = device
        self.tool_dir = tool_dir
        print(f"[*] SGCAgent initialized on device: {self.device}")

        self.working_memory = None
        self.ollama_config = {}

        
        try:
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                self.ollama_config = cfg.get("ollama", {})
        except Exception as e:
            print(f"[!] Config load error: {e}")

        self.tool_names = []
        self.tool_map = {}
        self.graph_manager = None
        self.retriever = None
        self.raw_embeddings = None
        self.attempt_tool_chain = []
        self.tool_set = OrderedDict()

    def get_text_embedding(self, text: str) -> torch.Tensor:

        try:
            url = self.ollama_config.get("embedding_url")
            model_name = self.ollama_config.get("model_name")
            data = {"model": model_name, "prompt": text}

            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()

            embedding_list = response.json()["embedding"]
            tensor = torch.tensor(embedding_list, dtype=torch.float32)
            tensor = tensor.unsqueeze(0)
            return tensor.to(self.device)

        except Exception as e:
            dim = self.ollama_config.get("embedding_dim", 768)
            print(f"Error getting embedding: {e}")
            return torch.zeros((1, dim), dtype=torch.float32, device=self.device)

    def load_trajectory_from_file(self, file_path, numbers: int):
        """
        工具图轨迹学习的接口
        """
        with open(file_path, 'r') as file:
            data = json.load(file)
        trajectories = list(data.values())
        trajectories_list = trajectories[:numbers]

        return trajectories_list

    def _assert_tool_index_alignment(self):
        
        for i, name in enumerate(self.retriever.tool_names):
            assert name in self.tool_map, f"[ALIGN ERROR] Tool '{name}' not in tool_map"
            assert self.tool_map[name] == i, (
                f"[ALIGN ERROR] tool_map mismatch: "
                f"tool_names[{i}]='{name}', "
                f"but tool_map['{name}']={self.tool_map[name]}"
            )

    def extract_all_json_blocks(self, text: str):
        """
        提取LLM输出的JSON块
        """
        results = []
        stack = []
        start = None

        in_str = False
        escape = False

        for i, ch in enumerate(text):
            if in_str:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue
            if ch in "{[":
                if not stack:
                    start = i
                stack.append(ch)

            elif ch in "}]":
                if not stack:
                    continue

                open_ch = stack.pop()
                if (open_ch == "{" and ch != "}") or (open_ch == "[" and ch != "]"):
                    stack = []
                    start = None
                    continue

                if not stack and start is not None:
                    block = text[start:i + 1]
                    try:
                        results.append(json.loads(block))
                    except Exception:
                        pass
                    start = None

        return results

    def _parse_json(self, text: str):
        """
        解析提取后的JSON块
        """
        blocks = self.extract_all_json_blocks(text)

        parsed = {
            "tasks": None,
            "tool_calls": None,
            "status": None,
            "reason": None,
        }

        for b in blocks:
            if (
                    isinstance(b, list) and b
                    and isinstance(b[0], dict)
                    and "step" in b[0]
            ):
                parsed["tasks"] = b
                continue

            if isinstance(b, dict):
                if "tool_calls" in b and isinstance(b["tool_calls"], list):
                    parsed["tool_calls"] = b["tool_calls"]
                
                if "reason" in b:
                    parsed["reason"] = b["reason"]
                
                if "status" in b:
                    parsed["status"] = b 
                continue

            if (
                    isinstance(b, list) and b
                    and isinstance(b[0], dict)
                    and "tool_name" in b[0]
            ):
                parsed["tool_calls"] = b
                continue

        return parsed

    def build_tool_pool(self, tasks: List[Dict], top_k: int = 5):
        """
        对一批 tasks 执行检索，构建工具池
        """
        print("\n>>> [Retrieval] (Re)Building tool pool from tasks ...")

        for task in tasks:
            tool_search = task.get("tool_search", "")
            action = task.get("action", "")
            query = task.get("query", "")

            search_vec = self.get_text_embedding(f"{action} {tool_search}")

            candidates = self.retriever.search(search_vec, top_k=top_k)
            if not candidates:
                print(f"[Warning] No tools found for task: {query}")
                continue

            task['suggested_tools'] = []
            for rank, c in enumerate(candidates):
                tool_name = c["name"]
                score = c["score"]

                task['suggested_tools'].append({
                    "name": tool_name,
                    "score": score,
                    "rank": rank + 1
                })

                if tool_name not in self.tool_set:
                    self.tool_set[tool_name] = {
                        "id": c["id"],
                        "name": c["name"],
                        "score": c["score"]
                    }
                else:
                    if score > self.tool_set[tool_name]["score"]:
                        self.tool_set[tool_name]["score"] = score                

        print(f"[Retrieval Done] Tool pool size: {len(self.tool_set)}")

    async def _decompose_query(self, query: str) -> List[Dict]:
        """
        任务分解
        """
        prompt = self.sys_prompt_template + DECOMPOSE_PROMPT.format(query=query)
        resp = await self._call_llm(prompt)
        tasks = self._parse_json(resp)
        tasks = tasks["tasks"]

        if not tasks or not isinstance(tasks, list):
            tasks = [{
                "step": 1,
                "action": "execute",
                "query": query
            }]
        self.history.append({"role": "assistant", "content": f"[Task Decompose]\n{tasks}"})
        # print("History is ", self.history)
        return tasks

    async def _tool_call(self, current_task_node: Dict) -> Dict:
        """
        工具调用：接收当前任务和候选工具列表
        """
        task_query = current_task_node['query']
        tool_candidates = list(self.tool_set.values())
        display_candidates = tool_candidates[:20]
        
        display_list = []
        for tool_data in display_candidates:
            name = tool_data['name']
            
            info = self.tool_registry.get_unified_tool_info(name)
            if not info:
                continue

            tool_schema = {
                "name": info["name"],
                "description": info["description"],
                "parameters": info["parameters"],
                "relevance_score": f"{tool_data['score']:.4f}" 
            }
            display_list.append(tool_schema)

        tools_info_str = json.dumps(display_list, indent=2)

        tool_context = self.working_memory.recent_steps

        prompt = self.sys_prompt_template + ACT_PROMPT.format(
            task_query=task_query,
            tools_info=tools_info_str,
            tool_context=tool_context,
            feed_back=self.working_memory.feedback_message
        )

        resp = await self._call_llm(prompt)

        parsed = self._parse_json(resp)

        self.history.append({
            "role": "assistant",
            "content": f"[Decision Logic]\n{json.dumps(parsed, ensure_ascii=False, indent=2)}"
        })

        return parsed

    async def _re_plan(self) -> List[Dict]:
        context = self.working_memory.get_final_report_view()
        prompt = self.sys_prompt_template + REPLAN_PROMPT.format(
            original_query=self.working_memory.original_query,
            finished_tasks=self.working_memory.finished_tasks,
            context=context,
            feed_back=self.working_memory.feedback_message
        )
        resp = await self._call_llm(prompt)
        parsed = self._parse_json(resp)
        tasks = parsed["tasks"]

        if not tasks or not isinstance(tasks, list):
            tasks = []

        self.history.append({
            "role": "assistant",
            "content": f"[RePlan]\n{json.dumps(tasks, ensure_ascii=False, indent=2)}"
        })

        return tasks
    
    #TODO
    # API-Bank need final summary to be generated by LLM, so we keep this function here for use.
    # async def _final_summary(self, user_query: str, choices: str) -> str:
    #     prompt = FINAL_SUMMARY_PROMPT.format(
    #         user_query=user_query,
    #         choices=choices,
    #         working_memory=self.working_memory.get_final_report_view()
    #     )
    #     print("\n>>> [Final Summary] Generating final answer...",prompt)
    #     final_summary = await self._call_llm(prompt=prompt)
    #     self.history.append({"role": "assistant", "content": final_summary})
    #     return final_summary
    