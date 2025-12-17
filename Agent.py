import re
import json
import requests
import yaml
import torch
from typing import Any, Optional, List, Dict
import inspect
from model import LLM
from Working_mem import WorkingMemory
from Toolregistry import ToolRegistry
from GraphManager import GraphManager
from SGCRetriever import SGCRetriever
from prompt import DECOMPOSE_PROMPT, ACTION_PROMPT, SYSTEM_PROMPT, REPLAN_PROMPT, JUDGER_PROMPT


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
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        super().__init__(initial_model, SYSTEM_PROMPT, "./outputs/outputs.jsonl")

        self.device = device
        print(f"[*] SGCAgent initialized on device: {self.device}")

        self.working_memory = None
        self.ollama_config = {}
        try:
            with open("/media/csudxy0218/ZL/AgentToolmem/config.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                self.ollama_config = cfg.get("ollama", {})
        except Exception as e:
            print(f"[!] Config load error: {e}")

        # 2. SGC 系统组件占位
        self.tool_names = []
        self.tool_map = {}  # name -> id
        self.graph_manager = None
        self.retriever = None
        self.raw_embeddings = None

        # 3. 轨迹缓冲区
        self.attempt_tool_chain = []

    def save_data(self, query: str, final_result: str, status: str = "success"):

        data = {
            "query": query,
            "status": status,
            "final_result": final_result,
            "history": list(self.history),
        }
        with open("/Earth-agent/data/outputs.jsonl", "a", encoding="utf-8") as f:
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

    def init_sgc_system(self):
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
        for desc in desc:
            # get_text_embedding 已经处理了 device
            raw_embeds_list.append(self.get_text_embedding(desc))

        if raw_embeds_list:
            # torch.cat 在 GPU 上执行拼接
            self.raw_embeddings = torch.cat(raw_embeds_list, dim=0)  # (N, D)
        else:
            self.raw_embeddings = torch.empty(0, device=self.device)

        # 3. 初始化 SGC 检索器
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

    def _parse_json(self, text: str) -> Any:
        """
        Robust JSON parser for LLM output.
        Extracts the first valid JSON array or object from text.
        """

        if not text or not isinstance(text, str):
            return None

        codeblock = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", text, re.S)
        if codeblock:
            try:
                return json.loads(codeblock.group(1))
            except Exception:
                pass

        array_match = re.search(r"\[\s*\{.*?\}\s*\]", text, re.S)
        if array_match:
            try:
                return json.loads(array_match.group())
            except Exception:
                pass

        object_match = re.search(r"\{.*?\}", text, re.S)
        if object_match:
            try:
                return json.loads(object_match.group())
            except Exception:
                pass

        # 4️⃣ 彻底失败
        return None

    async def decompose_query(self, query: str) -> List[Dict]:
        """
        Step 1: 任务分解
        将用户查询分解为带有明确 'action' 的子问题序列。
        """
        prompt = self.sys_prompt_template + DECOMPOSE_PROMPT.format(query=query)
        resp = await self._llm_generate_text(prompt)
        tasks = self._parse_json(resp)
        self.history.append({"role": "assistant", "content": f"[Task Decompose]\n{tasks}"})
        print("History is ", self.history)
        return tasks if isinstance(tasks, list) else [{"step": 1, "action": "execute", "query": query}]

    async def generate_tool_args(self, candidates: List[Dict], task_query: str) -> List[Dict]:
        """
        接收候选工具列表
        """
        # 1. 批量构建工具信息 (Name + Desc + Schema)
        tools_list = []
        for cand in candidates:
            name = cand['name']
            # 获取描述
            info = self.tool_registry.get_unified_tool_info(name)

            tools_list.append({
                "name": info["name"],
                "description": info["description"],
                "parameters": info["parameters"]
            })

        # 转为 JSON 字符串，塞入 Prompt
        tools_info_str = json.dumps(tools_list, indent=2)
        last_tool_answer = self.working_memory.tool_context

        prompt = ACTION_PROMPT.format(
            num_tools=len(tools_list),
            task_query=task_query,
            tools_info=tools_info_str,
            tool_context=last_tool_answer
        )

        resp = await self._llm_clean_tool(prompt)
        parsed = self._parse_json(resp)
        self.history.append({"role": "assistant", "content": f"[Tool Selection]\n{parsed}"})
        print("History is ", self.history)
        if not parsed: return []

        # Case 1: LLM directly returns a list of tool calls
        if isinstance(parsed, list):
            return parsed

        # Case 2: LLM returns a dict with "tool_calls"
        if isinstance(parsed, dict):
            return parsed.get("tool_calls", [])

        # Fallback
        return []

    async def verify(self, task_query: str, tool_name: str, args: Dict, result: str) -> Dict:
        prompt = JUDGER_PROMPT.format(
            task_query=task_query,
            tool_name=tool_name,
            tool_args=json.dumps(args),
            truncated_result=str(result)[:500]
        )
        resp = await self._llm_generate_text(prompt)
        res = self._parse_json(resp)
        self.history.append({"role": "assistant", "content": f"[Verify]\n{res}"})
        print("History is ", self.history)
        if not res: return {"status": "SUCCESS", "reason": "Auto-pass due to parse error"}
        return res

    async def re_plan(self, failure_reason: str) -> List[Dict]:
        self.attempt_tool_chain = []
        prompt = REPLAN_PROMPT.format(
            original_query=self.working_memory.original_query,
            finished_tasks=self.working_memory.finished_tasks,
            failed_steps=self.working_memory.recent_steps[-1],
            failure_reason=failure_reason
        )
        resp = await self._llm_generate_text(prompt)
        tasks = self._parse_json(resp)
        self.history.append({"role": "assistant", "content": f"[RePlan]\n{tasks}"})
        print("History is ", self.history)
        return tasks if isinstance(tasks, list) else []

    async def run(self, user_query: str):
        # 1. 初始化
        self.history = []

        if self.retriever is None or self.graph_manager is None:
            self.init_sgc_system()

        self._assert_tool_index_alignment()

        self.working_memory = WorkingMemory(original_query=user_query)
        self.history.append({"role": "user", "content": user_query})

        # 2. 初始规划
        print(f"\n>>> [Planning] Decomposing User Query...")
        task_queue = await self.decompose_query(user_query)
        print(f"    Initial Plan: {len(task_queue)} steps.")

        # 3. 任务执行循环
        task_idx = 0
        MAX_GLOBAL_RETRIES = 3
        global_retry_count = 0

        # 大循环：遍历任务队列
        while task_idx < len(task_queue):
            current_task = task_queue[task_idx]
            self.working_memory.start_task(current_task['query'])
            self.attempt_tool_chain = []
            print(f"\n=== Step {task_idx + 1}: {current_task['query']} ===")

            # 每次新任务开始前，初始化黑名单
            # [重要] 必须在这里重置，否则上一个任务排除的工具会影响这个任务
            bad_tools = []
            local_success = False
            last_verification_error = None
            LOCAL_RETRIES = 2

            # 局部重试循环 (Attempt Loop)
            # 使用 while 而不是 for，以便更灵活地控制 (比如 ToolMismatch 时不消耗 attempt 计数，或者单独计数)
            attempt = 0
            while attempt <= LOCAL_RETRIES:
                print(f"   [Attempt {attempt + 1}] Processing...")

                # A. 检索工具 (带黑名单)
                # search_vec = self.get_text_embedding(f"{current_task['query']}")
                search_vec = self.get_text_embedding(f"{current_task['action']} {current_task['query']}")

                # 带有上一个子任务影响的工具检索
                prev_tool_id = None
                if self.attempt_tool_chain:
                    prev_tool_id = self.attempt_tool_chain[-1]

                candidates = self.retriever.search(
                    search_vec,
                    top_k=5,
                    avoid_names=bad_tools,
                    pre_tool=prev_tool_id
                )

                if not candidates:
                    print("     [Error] No candidates found.")
                    break

                # B1. LLM 选择工具 (返回列表)
                tool_calls = await self.generate_tool_args(candidates, current_task['query'])

                if not tool_calls:
                    print("     [Error] No tools selected. Retrying...")
                    attempt += 1
                    continue

                print(f"     [Plan] Selected {len(tool_calls)} tools: {[t['tool_name'] for t in tool_calls]}")
                # --- 子循环：执行这一步的所有工具 ---
                step_tools_success = True
                last_verification_error = None
                failed_tool_name = None

                for t_idx, call in enumerate(tool_calls):
                    tool_name = call.get('tool_name')
                    args = call.get('arguments', {})

                    print(f"       -> [Sub-call {t_idx + 1}] {tool_name}")

                    # 执行
                    try:
                        if not self.tool_registry.get_tool(tool_name):
                            raise ValueError(f"Tool '{tool_name}' not found")
                        result = await self.call_tool(tool_name, args)
                    except Exception as e:
                        result = f"SystemException: {e}"
                        # 这里不直接 break，而是通过 verify 来统一判断失败

                    # 验证 (Verify)
                    verification = await self.verify(current_task['query'], tool_name, args, result)

                    if verification['status'] == 'SUCCESS':
                        # 1. 记录日志
                        self.working_memory.record_tool_success(task_idx + 1, tool_name, args, result)

                        # 2. 图记录
                        tool_id = self.tool_map.get(tool_name)
                        if tool_id is not None:
                            self.attempt_tool_chain.append(tool_id)

                    else:
                        # --- 单个工具失败 ---
                        step_tools_success = False
                        print(f"       [Fail] {tool_name}: {verification.get('reason')}")

                        # 保存错误现场，供外部重试逻辑使用
                        last_verification_error = verification
                        failed_tool_name = tool_name

                        self.working_memory.record_tool_failure(
                            current_task['query'],
                            tool_name,
                            verification.get("error_type"),
                            verification.get("reason")
                        )
                        # 轨迹回滚
                        self.attempt_tool_chain = []
                        # 中断子循环（多工具链路中只要一环断了，后面就没必要执行了）
                        break

                        # --- 子循环结束 ---

                # C. 判定本轮 Attempt 结果
                if step_tools_success:
                    local_success = True

                    if len(self.attempt_tool_chain) >= 2:
                        self.graph_manager.update_from_trajectory(self.attempt_tool_chain)
                    self.attempt_tool_chain = []

                    self.working_memory.finished_tasks.append(current_task['query'])
                    break  # 成功！跳出 attempt 循环，进入下一个 Task

                else:
                    # --- 失败处理与重试逻辑 ---
                    err_type = last_verification_error.get('error_type') if last_verification_error else "Unknown"

                    if err_type == 'ToolMismatch':
                        print(f"     [Correction] Tool '{failed_tool_name}' mismatch. Switching tool...")
                        if failed_tool_name: bad_tools.append(failed_tool_name)
                        attempt += 1
                        continue

                    elif err_type == 'ArgumentError':
                        print("     [Correction] Retrying with adjusted arguments...")
                        attempt += 1
                        continue

                    else:
                        print(f"     [Correction] Runtime error ({err_type}). Retrying...")
                        attempt += 1
                        continue

            # --- 局部 attempt 循环结束 ---

            # D. 决策：继续还是重规划？
            if local_success:
                task_idx += 1  # 推进到下一步
            else:
                # 彻底失败，开始重规划
                print(f"\n[!] Step Failed after {LOCAL_RETRIES} retries. Initiating Re-planning...")

                if global_retry_count >= MAX_GLOBAL_RETRIES:
                    print("[!] Max global retries reached. Task Failed.")
                    break

                # 获取失败原因，用于 Prompt
                fail_reason = "Unknown"
                if last_verification_error:
                    fail_reason = last_verification_error.get('reason', 'Unknown')

                failure_reason_str = f"Step '{current_task['query']}' failed. Last error: {fail_reason}"

                # 调用重规划
                new_sub_tasks = await self.re_plan(failure_reason_str)

                if new_sub_tasks:
                    print(f"    [Re-plan] Generated {len(new_sub_tasks)} new steps.")

                    # 更新任务队列：保留已完成的 + 新生成的
                    task_queue = task_queue[:task_idx] + new_sub_tasks

                    # task_idx 保持不变，指向新计划的第一个任务
                    global_retry_count += 1
                else:
                    print("[!] Re-planning failed to generate tasks. Aborting.")
                    break
        # 4. 最终总结
        final_prompt = f"Summarize the final result for: {user_query}\nBased on: {self.working_memory.get_prompt_view()}"
        final_summary = await self._llm_clean_tool(prompt=final_prompt, history=self.history)
        self.history.append({"role": "assistant", "content": final_summary})
        self.save_data(query=user_query, final_result=final_summary)
        return final_summary
