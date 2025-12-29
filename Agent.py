import json
import requests
import yaml
import torch
import inspect
from collections import OrderedDict
from typing import Any, Optional, List, Dict
from model import LLM
from Working_mem import WorkingMemory
from Toolregistry import ToolRegistry
from GraphManager import GraphManager
from SGCRetriever import SGCRetriever
from prompt import DECOMPOSE_PROMPT, ACTION_PROMPT, SYSTEM_PROMPT, REPLAN_PROMPT, SUBTASK_VERIFY_PROMPT


class BaseAgent:
    def __init__(self, initial_model: str, sys_prompt_template: str, output_dir: str):

        self.initial_model = initial_model
        self.sys_prompt_template = sys_prompt_template
        self.output_dir = output_dir
        self.llm = LLM(initial_model)

        self.tool_registry = ToolRegistry()
        self.tool_list = []
        self.history = []

    def _normalize_chunk(self, chunk: Any) -> str:
        if isinstance(chunk, dict):
            if chunk.get("type") == "error":
                raise RuntimeError(chunk.get("error", "unknown tool error"))
            return chunk.get("text", json.dumps(chunk, ensure_ascii=False))
        if isinstance(chunk, (bytes, bytearray)):
            return chunk.decode("utf-8", errors="ignore")
        return str(chunk)

    async def call_tool(self, tool_name: str, arguments: Optional[dict] = None):
        arguments = arguments or {}
        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return f"Tool '{tool_name}' not found."

        fn = tool["callable"]

        try:
            out = fn(**arguments)
            if inspect.isawaitable(out):
                out = await out

            if hasattr(out, "__aiter__"):
                acc = ""
                async for chunk in out:
                    acc += self._normalize_chunk(chunk)
                text = acc

            elif hasattr(out, "__iter__") and not isinstance(out, (str, bytes, dict)):
                acc = ""
                for chunk in out:
                    acc += self._normalize_chunk(chunk)
                text = acc

            else:
                text = self._normalize_chunk(out)

            # 记录工具调用轨迹
            self.tool_list.append(tool_name)

            # 写入历史
            self.history.append({
                "role": "tool",
                "name": tool_name,
                "content": text
            })

            return text

        except Exception as e:
            return f"Exception calling tool {tool_name}: {e}"

    async def chat(self, prompt, llm_name=None, image_path=None):

        if llm_name:
            self.llm = LLM(llm_name)

        # 写入用户消息（保持最简单历史）
        self.history.append({"role": "user", "content": prompt})

        # 流式输出
        result = None
        acc = []

        async for chunk in self.llm.generate_stream_res(
            prompt=prompt,
            history=self.history,
            image_path=image_path
        ):
            if chunk.get("type") == "error":
                return f"Error: {chunk.get('error')}"

            if chunk.get("type") in ("text", "stream"):
                acc.append(chunk.get("text", ""))

            if chunk.get("type") == "final":
                result = chunk.get("text", "")

        result = result if result is not None else "".join(acc)

        self.history.append({"role": "assistant", "content": result})

        return result


class SGCAgent(BaseAgent):
    def __init__(self,
                 initial_model: str,
                 output_dir: str,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        self.output_dir = output_dir
        super().__init__(initial_model, SYSTEM_PROMPT, self.output_dir)
        self.device = device
        print(f"[*] SGCAgent initialized on device: {self.device}")

        self.working_memory = None
        self.ollama_config = {}
        self.yaml_path = "/media/csudxy0218/ZL/AgentToolmem/config.yaml"
        try:
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                self.ollama_config = cfg.get("ollama", {})
        except Exception as e:
            print(f"[!] Config load error: {e}")

        # 2. SGC 系统组件占位
        self.tool_names = []
        self.tool_map = {}
        self.graph_manager = None
        self.retriever = None
        self.raw_embeddings = None

        # 3. 轨迹缓冲区
        self.attempt_tool_chain = []

        # 4. 工具检索集合
        self.tool_set = OrderedDict()

    def save_data(self, query: str, final_result: str, status: str = "success"):

        data = {
            "query": query,
            "status": status,
            "final_result": final_result,
            "history": list(self.history),
        }
        with open(self.output_dir, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        print(f"[*] Task archived to {self.output_dir}")

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
        从文件中读取历史轨迹，返回工具名称的列表
        """
        with open(file_path, 'r') as file:
            data = json.load(file)
        trajectories = list(data.values())
        trajectories_list = trajectories[:numbers]

        return trajectories_list

    def init_sgc_system(self, trajectory_file_path=None):
        """
        初始化图检索系统。
        注意：必须在 ToolRegistry 注册完所有工具后调用此方法。
        """
        tools = self.tool_registry.tools  # 获取字典 {name: tool_conf}
        if not tools:
            print("[!] No tools registered in ToolRegistry.")
            return

        self.tool_names = list(tools.keys())
        # 提取工具描述用于初始化向量
        desc = [
            self.tool_registry.get_unified_tool_info(name)["description"]
            for name in self.tool_names
        ]
        self.tool_map = {name: i for i, name in enumerate(self.tool_names)}

        num_nodes = len(self.tool_names)

        print(f"[*] Initializing SGC Graph for {num_nodes} tools...")

        # 1. 初始化图管理器 (传入 device，矩阵将创建在 GPU)
        # 初始状态：离散节点，无边
        self.graph_manager = GraphManager(num_nodes=num_nodes, device=self.device)

        # 2. 批量生成初始工具嵌入 (Raw Embeddings)
        raw_embeds_list = []
        for des in desc:
            # get_text_embedding 已经处理了 device
            raw_embeds_list.append(self.get_text_embedding(des))

        if raw_embeds_list:
            # torch.cat 在 GPU 上执行拼接
            self.raw_embeddings = torch.cat(raw_embeds_list, dim=0)  # (N, D)
        else:
            self.raw_embeddings = torch.empty(0, device=self.device)
        # 历史轨迹输入
        if trajectory_file_path:
            # print("[*] Reading trajectory from file and updating graph...")
            trajectories = self.load_trajectory_from_file(trajectory_file_path, 50)
            # print(f"[*] Reading trajectory from {trajectories}")

            total_edges = 0
            total_missing = 0

            for traj in trajectories:  # 每一条工具链
                indices = []
                last = None

                # print("\n[DEBUG] traj =", traj)
                # print("[DEBUG] type(traj) =", type(traj))

                for i, x in enumerate(traj):
                    print(f"   [DEBUG] traj[{i}] =", repr(x), "type:", type(x))

                for name in traj:  # 每一个工具名
                    if not isinstance(name, str):
                        continue

                    if name not in self.tool_map:
                        total_missing += 1
                        continue

                    idx = self.tool_map[name]

                    # 去掉连续重复（避免自环）
                    if last is not None and idx == last:
                        continue

                    indices.append(idx)
                    last = idx

                if len(indices) >= 2:
                    self.graph_manager.update_from_trajectory(indices)
                    total_edges += len(indices) - 1

            print(f"[*] Trajectory init done: edges={total_edges}, missing_tools={total_missing}")
        # 4. 初始化 SGC 检索器
        # 计算将利用 GPU 加速
        self.retriever = SGCRetriever(
            graph_manager=self.graph_manager,
            tool_names=self.tool_names,
            raw_embeddings=self.raw_embeddings,
            alpha=0.2  # 调节 SGC 聚合强度
        )
        print("[*] SGC System ready.")

    def _assert_tool_index_alignment(self):
        # tool_names[i] <-> tool_map[name] == i
        for i, name in enumerate(self.retriever.tool_names):
            assert name in self.tool_map, f"[ALIGN ERROR] Tool '{name}' not in tool_map"
            assert self.tool_map[name] == i, (
                f"[ALIGN ERROR] tool_map mismatch: "
                f"tool_names[{i}]='{name}', "
                f"but tool_map['{name}']={self.tool_map[name]}"
            )

    async def _llm_generate_text(self, prompt: str, history: List[Dict] = None) -> str:
        """辅助方法：非流式获取 LLM 完整响应"""
        acc = []
        async for chunk in self.llm.generate_stream_res(prompt=prompt, history=history):
            if chunk.get("type") in ("text", "stream", "final"):
                acc.append(chunk.get("text", ""))
        return "".join(acc).strip()

    async def _llm_clean_tool(self, prompt: str, history: List[Dict] = None) -> str:
        acc = []
        async for chunk in self.llm.generate_stream_res(prompt=prompt, history=history):
            if chunk.get("type") == "final":
                acc.append(chunk.get("text", ""))
        return "".join(acc).strip()

    def extract_all_json_blocks(self, text: str):
        """
        从文本中提取所有合法 JSON（object 或 array）
        返回 List[Any]
        """
        results = []
        stack = []
        start = None

        in_str = False
        escape = False

        for i, ch in enumerate(text):
            # 处理字符串状态（忽略字符串内部的 { } [ ]）
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

            # 只在非字符串状态下处理括号
            if ch in "{[":
                if not stack:
                    start = i
                stack.append(ch)

            elif ch in "}]":
                if not stack:
                    continue

                open_ch = stack.pop()

                # 可选：括号类型匹配检查（更严格）
                if (open_ch == "{" and ch != "}") or (open_ch == "[" and ch != "]"):
                    # 不匹配：清空，避免产生错误块
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
        blocks = self.extract_all_json_blocks(text)

        parsed = {
            "tasks": None,
            "tool_calls": None,
            "verification": None,
            "reason": None,
        }

        for b in blocks:
            # 1. 解析任务列表 (Planning)
            if (
                    isinstance(b, list) and b
                    and isinstance(b[0], dict)
                    and "step" in b[0]
            ):
                parsed["tasks"] = b
                continue

            # 2. 解析工具调用 (Action) - 格式1: {tool_calls: [...]}
            if isinstance(b, dict) and "tool_calls" in b and isinstance(b["tool_calls"], list):
                parsed["tool_calls"] = b["tool_calls"]
                if "reason" in b:
                    parsed["reason"] = b["reason"]
                continue

            # 3. 解析工具调用 (Action) - 格式2: [{tool_name: ...}]
            if (
                    isinstance(b, list) and b
                    and isinstance(b[0], dict)
                    and "tool_name" in b[0]
            ):
                parsed["tool_calls"] = b
                continue

            # 4. 解析验证结果 (Verification)
            if isinstance(b, dict) and "status" in b:
                if "error_type" in b or "reason" in b:
                    parsed["verification"] = b
                    continue

        return parsed

    def update_tool_pool_with_children(self):
        print("\n>>> [ToolPool Update] Expanding tool pool with child nodes...")

        if not self.attempt_tool_chain:
            return

        new_tool_pool = OrderedDict()

        parent_id = self.attempt_tool_chain[-1]
        child_ids = self.graph_manager.get_child_tools(parent_id)

        for cid in child_ids:
            tool_name = self.tool_names[cid]
            if tool_name is None:
                continue

            if tool_name not in self.tool_set:
                new_tool_pool[tool_name] = {
                    "id": cid,
                    "name": tool_name,
                    "vec": None,
                }
                print(f"New tools {tool_name} is appended!")

        for name, info in self.tool_set.items():
            if name not in new_tool_pool:
                new_tool_pool[name] = info

        self.tool_set = new_tool_pool

    def build_tool_pool(self, tasks: List[Dict], top_k: int = 5):
        """
        对一批 tasks 执行检索，增量更新 self.tool_set（不包含 score）
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

                # 将排名信息存入 task 结构
                task['suggested_tools'].append({
                    "name": tool_name,
                    "score": score,
                    "rank": rank + 1  # 1-based ranking
                })

                # 更新全局 tool_set
                if tool_name not in self.tool_set:
                    self.tool_set[tool_name] = {
                        "id": c["id"],
                        "name": c["name"],
                        "vec": c["vec"],
                    }

        print(f"[Retrieval Done] Tool pool size: {len(self.tool_set)}")

    async def _decompose_query(self, query: str) -> List[Dict]:
        """
        Step 1: 任务分解
        将用户查询分解为带有明确 'action' 的子问题序列。
        """
        prompt = self.sys_prompt_template + DECOMPOSE_PROMPT.format(query=query)
        resp = await self._llm_generate_text(prompt)
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

    async def _generate_tool_args(self, current_task_node: Dict, all_tools: List[Dict]) -> List[Dict]:
        """
        接收候选工具列表
        """
        task_query = current_task_node['query']
        # 获取 SGC 检索到的工具列表 (已经按分数排序)
        suggested_tools = current_task_node.get('suggested_tools', [])

        display_list = []
        added_names = set()

        # === 1. 优先放入检索到的工具 (按 SGC 分数降序) ===
        for item in suggested_tools:
            name = item['name']

            # 获取工具完整信息
            info = self.tool_registry.get_unified_tool_info(name)
            if not info:
                continue

            # 直接加入列表，不添加任何额外的 "rank" 或 "recommendation" 字段
            display_list.append({
                "name": info["name"],
                "description": info["description"],
                "parameters": info["parameters"]
            })
            added_names.add(name)

        # === 2. 补充全局池中的其他工具 ===
        for tool_data in all_tools:
            name = tool_data['name']

            if name not in added_names:
                info = self.tool_registry.get_unified_tool_info(name)
                if not info:
                    continue

                display_list.append({
                    "name": info["name"],
                    "description": info["description"],
                    "parameters": info["parameters"]
                })
                added_names.add(name)

        tools_info_str = json.dumps(display_list, indent=2)
        # print(f"Here are all Tools\n SubTask{task_query}", tools_info_str)
        tool_context = self.working_memory.recent_steps
        print(tool_context)

        prompt = self.sys_prompt_template + ACTION_PROMPT.format(
            num_tools=len(display_list),
            task_query=task_query,
            tools_info=tools_info_str,
            tool_context=tool_context
        )

        resp = await self._llm_clean_tool(prompt)
        # print("Here is resp", resp)
        parsed = self._parse_json(resp)
        self.history.append({"role": "assistant", "content": f"[Tool Selection]\n{parsed}"})
        # print("History is ", self.history)
        tool_calls = parsed["tool_calls"]

        if not tool_calls:
            print("No tool_calls found in LLM output.")
            return []
        return tool_calls

    async def _subtask_verify(self, attempt_outputs: List[Dict]) -> Dict:
        current_task = self.working_memory.current_task
        results_str = json.dumps(attempt_outputs, ensure_ascii=False, indent=2)

        prompt = self.sys_prompt_template + SUBTASK_VERIFY_PROMPT.format(
            subtask=current_task,
            subtask_results=results_str
        )

        resp = await self._llm_generate_text(prompt)
        parsed = self._parse_json(resp)

        # 鲁棒性处理：防止解析失败
        verification = parsed.get("verification")
        if not verification:
            # 如果解析不到，尝试找整个 JSON
            verification = parsed if "status" in parsed else None

        if not verification:
            return {"status": "FAILURE", "reason": "Failed to parse verification JSON", "suggestion": "Retry"}

        self.history.append({
            "role": "assistant",
            "content": f"[Subtask_verify]\n{verification}"
        })
        return verification

    async def _re_plan(self) -> List[Dict]:
        context = self.working_memory.get_final_report_view()
        prompt = self.sys_prompt_template + REPLAN_PROMPT.format(
            original_query=self.working_memory.original_query,
            finished_tasks=self.working_memory.finished_tasks,
            context=context
        )
        resp = await self._llm_generate_text(prompt)
        parsed = self._parse_json(resp)
        tasks = parsed["tasks"]

        if not tasks or not isinstance(tasks, list):
            tasks = []

        self.history.append({
            "role": "assistant",
            "content": f"[RePlan]\n{json.dumps(tasks, ensure_ascii=False, indent=2)}"
        })

        # print("History is ", self.history)

        return tasks

    async def run(self, user_query: str, choices):
        # 1. 初始化组件
        self.history = []
        self.tool_set = OrderedDict()
        self.attempt_tool_chain = []

        # TODO 历史工具路径
        trajectory_path = "/media/csudxy0218/ZL/AgentToolmem/Earth-agent/GT/tool_chain.json"

        if self.retriever is None or self.graph_manager is None:
            if trajectory_path is None:
                self.init_sgc_system()
            else:
                self.init_sgc_system(trajectory_path)

        self._assert_tool_index_alignment()

        self.working_memory = WorkingMemory(original_query=user_query)
        self.history.append({"role": "user", "content": user_query})

        # 2. 初始规划
        print(f"\n>>> [Planning] Decomposing User Query...")
        task_queue = await self._decompose_query(user_query)
        print(f"    Initial Plan: {len(task_queue)} steps.")

        # 3. 检索工具池
        print("\n>>> [Retrieval] Collecting tools for ALL planned tasks...")
        self.build_tool_pool(task_queue)

        # 4. 任务执行循环
        task_idx = 0
        MAX_GLOBAL_RETRIES = 3
        global_retry_count = 0

        while task_idx < len(task_queue):
            current_task = task_queue[task_idx]
            self.working_memory.start_task(current_task['query'])

            print(f"\n=== Step {task_idx + 1}: {current_task['query']} ===")

            step_success = False
            local_retries = 2
            attempt = 0

            while attempt <= local_retries:
                print(f"   [Attempt {attempt + 1}/{local_retries + 1}] Processing...")

                tool_calls = await self._generate_tool_args(current_task, list(self.tool_set.values()))

                if not tool_calls:
                    # 1. 检查上一关 (Global History)
                    has_global = len(self.working_memory.global_history) > 0
                    last_global = self.working_memory.global_history[-1] if has_global else None

                    # 2. [新增] 检查当前关的上一轮 (Recent Steps)
                    # 如果刚才跑过工具且成功了，但 LLM 现在不调工具，说明它忽略了 Verify 的反馈
                    has_recent = len(self.working_memory.recent_steps) > 0
                    last_recent = self.working_memory.recent_steps[-1] if has_recent else None

                    if (last_global and
                            last_global.get("status") == "SUCCESS" and
                            last_global.get("current_task") == current_task["query"]):
                        step_success = True
                        break

                    # [新增] 针对“验证失败后 LLM 罢工”的处理
                    elif (last_recent and
                          last_recent.get("status") == "SUCCESS" and
                          last_recent.get("current_task") == current_task["query"]):

                        print(f"     [Warning] Agent refuses to act after verification failure.")
                        # 策略：强制给一条系统反馈，告诉它“别偷懒，刚才的验证没过”
                        self.working_memory.add_feedback_message(
                            "System Alert: You decided NOT to call tools, but the previous verification FAILED. "
                            "You must modify your arguments or try a different tool to satisfy the goal."
                        )
                        attempt += 1
                        continue

                    else:
                        print(f"     [Warning] No tools selected. (Attempt {attempt + 1})")
                        attempt += 1
                        continue

                print(f"     [Plan] Selected {len(tool_calls)} tools: {[t['tool_name'] for t in tool_calls]}")

                current_attempt_tools_ok = True
                last_verification_error = None
                current_attempt_outputs = []

                for t_idx, call in enumerate(tool_calls):
                    tool_name = call.get('tool_name')
                    args = call.get('arguments', {})

                    print(f"       -> [Sub-call {t_idx + 1}] {tool_name}")
                    execution_status = "SUCCESS"
                    error_msg = None

                    try:
                        if not self.tool_registry.get_tool(tool_name):
                            raise ValueError(f"Tool '{tool_name}' not found")
                        result = await self.call_tool(tool_name, args)

                    except Exception as e:
                        result = f"SystemException: {e}"
                        execution_status = "FAIL"
                        error_msg = str(e)

                    if execution_status == "SUCCESS":
                        print(f"     [Action Success] {tool_name}")
                        self.working_memory.record_tool_success(
                            step=task_idx + 1,
                            tool=tool_name,
                            args=args,
                            result=result)

                        current_attempt_outputs.append({
                            "tool": tool_name,
                            "args": args,
                            "result": str(result)
                        })

                        tool_id = self.tool_map.get(tool_name)
                        if tool_id is not None:
                            if self.attempt_tool_chain:
                                prev_tool_id = self.attempt_tool_chain[-1]
                                prev_tool = self.tool_names[prev_tool_id]
                                if prev_tool_id != tool_id:
                                    self.graph_manager.add_edge(prev_tool_id, tool_id, weight=1.0)
                                    print(f"Update SGC graph from {prev_tool} to {tool_name}")
                            self.attempt_tool_chain.append(tool_id)

                    else:
                        current_attempt_tools_ok = False
                        print(f"       [Fail] {tool_name}: {error_msg}")

                        # 构造一个假的 verification 对象方便下面统一处理
                        last_verification_error = {
                            "error_type": "ToolExecutionError",
                            "reason": error_msg
                        }
                        self.attempt_tool_chain = []
                        self.working_memory.record_tool_failure(
                            step=task_idx + 1,
                            tool=tool_name,
                            error_type="ToolExecutionError",
                            reason=error_msg
                        )
                        break

                # C. 判断本轮尝试结果
                if current_attempt_tools_ok:
                    print(f"     [Verify Task] checking results...")
                    task_verification = await self._subtask_verify(current_attempt_outputs)

                    if task_verification['status'] == 'SUCCESS':
                        print(f"     [Task Success] {task_verification['reason']}")
                        step_success = True
                        break
                    else:
                        # 工具跑通了，但任务没完成 (例如：文件列表为空，计算结果为NaN)
                        print(f"     [Task Fail]\n[Reason] {task_verification.get('reason')}")
                        print(f"     [Suggestion] {task_verification.get('suggestion')}")

                        self.attempt_tool_chain = []
                        self.working_memory.add_feedback_message(
                            f"Verification Failed: {task_verification.get('reason')}. Suggestion: {task_verification.get('suggestion')}"
                        )
                else:
                    err_type = last_verification_error.get('error_type') if last_verification_error else "Unknown"
                    print(f"     [Retry Trigger] Failure reason: {err_type}. Retrying...")
                    attempt += 1

            if step_success:
                print(f"   [Success] Step {task_idx + 1} completed.")
                self.working_memory.finished_tasks.append(current_task['query'])
                # TODO 更新子节点到工具池
                # if task_idx == 0:
                #     self.update_tool_pool_with_children()
                task_idx += 1
            else:
                # D. 任务失败，触发 Re_plan
                print(f"[!] Step {task_idx + 1} failed after {local_retries + 1} attempts.")

                if global_retry_count < MAX_GLOBAL_RETRIES:
                    print("    >>> Triggering Re-plan...")
                    new_sub_tasks = await self._re_plan()

                    if new_sub_tasks:
                        print(f"    [Re-plan Success] Generated {len(new_sub_tasks)} new steps.")

                        task_queue = task_queue[:task_idx] + new_sub_tasks
                        self.build_tool_pool(new_sub_tasks)

                        global_retry_count += 1
                        continue
                    else:
                        print("[!] Re-planning returned empty tasks. Aborting.")
                        break
                else:
                    print("[!] Max global retries reached. Task Failed.")
                    break

        # 最终总结
        final_prompt = f"""
            You are generating the final report for this agent run.
            User query:
            {user_query}
            
            Answer Choices:
            {choices}

            Working memory view:
            {self.working_memory.get_final_report_view()}

            Requirements:
            1) **Always output a final answer from the available answer choices (A/B/C/D)**, if provided in the user query.
            2) **If the task FAILED**, include the following:
               - Completed steps: List out all the steps that were executed.
               - Last failure information: Provide details on what went wrong (e.g., incorrect input, tool malfunction, etc.).
               - What is missing: Clearly state what part of the task could not be completed and why.

            3) **When answer choices (A/B/C/D) are provided in the user query**, select the best option from the available outputs. The answer should match **the tool’s execution results**. Ensure you check the tool output carefully for correctness before selecting the answer. If no answer matches, indicate that it cannot be determined from the available tool outputs.
            4) **Verification**: If the tool output contains multiple potential answers or ambiguous results, state that it is unclear and cannot be definitively answered from the available tool outputs. Always ensure the correct matching between the question's requirements and the tool’s results.

            **Rely on the information, only output (A/B/C/D)** 
            """
        final_summary = await self._llm_clean_tool(prompt=final_prompt)
        self.history.append({"role": "assistant", "content": final_summary})
        self.save_data(query=user_query, final_result=final_summary)
        return final_summary

    # def print_graph_edges(self, graph_manager, tool_names, k=20):
    #     edges = torch.nonzero(graph_manager.adj, as_tuple=False)
    #
    #     print(f"[CHECK] Showing {min(k, edges.shape[0])} edges (parent → child):")
    #     for i in range(min(k, edges.shape[0])):
    #         p, c = edges[i].tolist()
    #         w = graph_manager.adj[p, c].item()
    #         print(f"  {tool_names[p]}  →  {tool_names[c]}   (weight={w})")
